"""Two-node GLM HTTP serving with native GLM templates and fixed-depth MTP."""
import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path
import sys
import time

from .qwen_engine import QwenClusterEngine


def load_http():
    for name in ('image_cache', 'serve_metrics', 'serve_ping', 'serve_reqlog'):
        sys.modules.setdefault(name, importlib.import_module('tools.strix_halo.' + name))
    return importlib.import_module('tools.glm.serve')


class GlmClusterEngine(QwenClusterEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_draft = self.ndt
        self.heartbeats = 0

    def probe(self):
        # Serialize with whole GPU transactions; never interleave channel frames.
        with self.core.lock:
            if self.core.closed:
                return
            try:
                for client in self.core.pipeline.clients:
                    if hasattr(client, 'ping') and client.ping() != {'alive': True}:
                        raise RuntimeError('invalid worker heartbeat')
                self.heartbeats += 1
            except BaseException:
                self.core._fail()
                raise

    async def generate(self, prompt, *, max_tokens, temperature, top_p, stop,
                       img_stats=None, **kwargs):
        async for delta in super().generate(prompt, max_tokens=max_tokens,
                sampling={'temperature': temperature}, stop=stop, **kwargs):
            yield delta


def create_app(engine, model_id, template):
    from aiohttp import web
    http = load_http()
    http.SERVE_DEFAULTS.update(temperature=0.0, top_p=1.0, max_tokens=4096)
    app = http.create_app(engine, model_id, template)

    @web.middleware
    async def capabilities(request, handler):
        if engine.core.closed or engine.core.pipeline.failed:
            return web.json_response({'error': {'message': 'Cluster workers require restart'}}, status=503)
        if request.method == 'POST' and request.path in ('/v1/chat/completions', '/v1/completions', '/completion'):
            try:
                body = await request.json()
            except (ValueError, TypeError):
                return await handler(request)
            if isinstance(body, dict):
                unsupported = [k for k in ('logprobs', 'top_logprobs', 'thinking_budget', 'loop_guard') if body.get(k)]
                if body.get('temperature') not in (None, 0):
                    unsupported.append('sampled decoding')
                if body.get('speculative') is False:
                    unsupported.append('plain decoding')
                if unsupported:
                    return web.json_response({'error': {'message': 'Cluster currently supports greedy MTP only; unsupported: ' + ', '.join(unsupported)}}, status=400)
        response = await handler(request)
        if request.method == 'GET' and request.path in ('/health', '/v1/models') and response.status == 200:
            data = json.loads(response.text)
            geometry = dict(max_context=engine.ctx, sessions=engine.sessions,
                            cache_tokens=engine.ctx * engine.sessions, clustered=True, greedy_only=True,
                            worker_heartbeats=engine.heartbeats)
            if request.path == '/v1/models':
                for item in data['data']:
                    item.update(geometry)
            else:
                data.update(geometry)
            return web.json_response(data)
        return response

    app.middlewares.insert(0, capabilities)
    heartbeat = {}
    async def maintain(_):
        async def run():
            while not engine.core.closed:
                await asyncio.to_thread(engine.probe)
                await asyncio.sleep(15)
        heartbeat['task'] = asyncio.create_task(run())
    async def close(_):
        heartbeat['task'].cancel()
        await asyncio.gather(heartbeat['task'], return_exceptions=True)
        await asyncio.gather(*app['worker_tasks'], return_exceptions=True)
        await engine.close()
    app.on_startup.append(maintain)
    app.on_cleanup.append(close)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'weights-manifest', 'env', 'token-file'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--peer', default='10.203.0.2')
    ap.add_argument('--peer-port', type=int, default=9034)
    ap.add_argument('--context', type=int, default=262144)
    ap.add_argument('--sessions', type=int, default=2)
    ap.add_argument('--depth', type=int, default=2, choices=(1, 2, 3))
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8731)
    ap.add_argument('--model-id', default='glm-5.3-flash-yamz')
    args = ap.parse_args()
    from .session import Sessions
    Sessions(args.context, args.sessions)
    if not 8 <= args.chunk <= 2048:
        ap.error('chunk must be8..2048')
    for k, v in json.loads(Path(args.env).read_text()).items():
        if k not in ('EXL3_ROOT', 'EXL3_VENV', 'EXL3_ROCM_SDK'):
            os.environ[k] = v
    path = Path(args.token_file)
    stat = path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        ap.error('worker token must be private and user-owned')
    token = bytes.fromhex(path.read_text().strip())
    if len(token) < 32:
        ap.error('worker token must have at least32 random bytes')
    import torch
    from aiohttp import web
    from exllamav3 import Config, Tokenizer
    from .generation import GreedyEngine
    from .speculative import GLM_MODE, SpeculativePipeline
    from .stage import load_stage
    from .worker import StageClient
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.94, 0)
    config = Config.from_directory(args.model)
    tokenizer = Tokenizer.from_config(config)
    stage = load_stage(args.model, 0, 26, 'cuda:0', args.weights_manifest,
        max_rows=args.chunk, cache_context=args.context, cache_sessions=args.sessions,
        cache_bits=8, experimental_glm=True, speculation_depth=args.depth)
    class Local:
        manifest = stage.manifest
        def call(self, op, values): return stage.execute(op, values)
        def close(self): pass
    pipe = None
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                remote = StageClient(args.peer, args.peer_port, token, timeout=900)
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline: raise
                time.sleep(1)
        pipe = SpeculativePipeline([Local(), remote], mode=GLM_MODE)
        core = GreedyEngine(pipe, vocab_size=tokenizer.actual_vocab_size,
                            prefill_chunk=args.chunk, adaptive_drafting=False)
        engine = GlmClusterEngine(core, tokenizer, config.eos_token_id_list)
        http = load_http()
        template = http.DEFAULT_TEMPLATE.read_text()
        app = create_app(engine, args.model_id, template)
        print(json.dumps(dict(cluster_ready=True, context=args.context, sessions=args.sessions,
            cache_tokens=args.context*args.sessions, allocated_bytes=torch.cuda.memory_allocated(),
            source=stage.manifest['source_id'])), flush=True)
        web.run_app(app, host=args.host, port=args.port)
    finally:
        if pipe: pipe.close()
        stage.model.unload()


if __name__ == '__main__':
    main()
