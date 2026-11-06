"""Two-node DeepSeek V4.1 with native chat, tool parsing and DSpark."""
import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path
import sys
import time
from .serve_glm import GlmClusterEngine


def load_http():
    for name in ('image_cache', 'serve_metrics', 'serve_ping', 'serve_reqlog'):
        sys.modules.setdefault(name, importlib.import_module('tools.strix_halo.' + name))
    return importlib.import_module('tools.ds41.serve')


class DeepSeekClusterEngine(GlmClusterEngine):
    supports_sampling = True
    def prompt_ids(self, prompt):
        return self.tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True).flatten().tolist()

    def count_tokens(self, text):
        return len(self.prompt_ids(text))

    async def generate(self, prompt, *, temperature=1.0, top_p=0.95, prompt_ids=None, **kwargs):
        if prompt_ids is None: prompt_ids = self.prompt_ids(prompt)
        from .qwen_engine import QwenClusterEngine
        async for delta in QwenClusterEngine.generate(self, prompt, prompt_ids=prompt_ids,
                sampling=dict(temperature=temperature, top_p=top_p), **kwargs):
            yield delta


def create_app(engine, model_id):
    from aiohttp import web
    http = load_http()
    app = http.create_app(engine, model_id, http.Defaults(temperature=1.0, top_p=0.95, max_tokens=16384))

    @web.middleware
    async def capabilities(request, handler):
        if engine.core.closed or engine.core.pipeline.failed:
            return web.json_response({'error': {'message': 'Cluster requires restart'}}, status=503)
        if request.method == 'POST' and request.path in ('/v1/chat/completions', '/v1/completions', '/completion'):
            try: body = await request.json()
            except (ValueError, TypeError): return await handler(request)
            if isinstance(body, dict):
                from .sampling import validate_sampling
                try:
                    validate_sampling(body.get('temperature', 1.0), body.get('top_p', .95), body.get('seed'))
                    for key in ('speculative', 'repetition_guard'):
                        if key in body and type(body[key]) is not bool:
                            raise ValueError(key + ' must be boolean')
                except ValueError as exc:
                    return web.json_response({'error': {'message': str(exc)}}, status=400)
        response = await handler(request)
        if request.method == 'GET' and request.path in ('/health', '/v1/models') and response.status == 200:
            data = json.loads(response.text)
            geometry = dict(max_context=engine.ctx, context_window=engine.ctx, sessions=engine.sessions,
                cache_tokens=engine.ctx * engine.sessions, clustered=True, greedy_only=False, default_temperature=1.0, default_top_p=.95, repetition_guard=True,
                worker_heartbeats=engine.heartbeats, draft_depth=engine.ndt,
                prefix_cache=engine.core.prefix_pool is not None,
                prefix_stats=engine.core.prefix_pool.stats() if engine.core.prefix_pool else None,
                adaptive_mtp=engine.core.policy_factory is not None,
                prefill_pipeline=hasattr(engine.core,"async_step"),
                two_session_overlap=hasattr(engine.core,"async_step"))
            if request.path == '/v1/models':
                for item in data['data']: item.update(geometry)
            else: data.update(geometry)
            return web.json_response(data)
        return response

    app.middlewares.insert(0, capabilities)
    tasks = {}
    async def start(_):
        async def heartbeat():
            while not engine.core.closed:
                await asyncio.to_thread(engine.probe)
                await asyncio.sleep(15)
        tasks['heartbeat'] = asyncio.create_task(heartbeat())
    async def close(_):
        tasks['heartbeat'].cancel()
        await asyncio.gather(tasks['heartbeat'], *app['worker_tasks'], return_exceptions=True)
        await engine.close()
    app.on_startup.append(start)
    app.on_cleanup.append(close)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'weights-manifest', 'token-file', 'env'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--peer', default='10.203.0.2')
    ap.add_argument('--peer-port', type=int, default=9034)
    ap.add_argument('--context', type=int, default=262144)
    ap.add_argument('--sessions', type=int, default=2)
    ap.add_argument('--depth', type=int, choices=range(1, 6), default=4)
    ap.add_argument('--draft-depth', type=int, choices=range(1,6), help='serving depth within the owned capacity')
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--max-rows', type=int, help='owned prefill capacity, separate from selected chunk')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8731)
    ap.add_argument('--model-id', default='deepseek-v4.1-flash-1017-2.68bpw')
    ap.add_argument('--qualify', help='run bounded one-load frontier before serving, save receipts here')
    ap.add_argument('--optimize', help='run cache, prefill and MTP gates before serving')
    ap.add_argument('--diagnose', help='run isolated teacher-forced diagnostics and exit without serving')
    args = ap.parse_args()
    for k, v in json.loads(Path(args.env).read_text()).items():
        if k not in ('EXL3_ROOT', 'EXL3_VENV', 'EXL3_ROCM_SDK'): os.environ[k] = v
    p = Path(args.token_file); stat = p.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077: ap.error('worker token must be private/user-owned')
    token = bytes.fromhex(p.read_text().strip())
    if len(token) < 32: ap.error('worker token needs at least32 random bytes')
    import torch
    from aiohttp import web
    from exllamav3 import Config, Tokenizer
    from .ds41_stage import load_stage
    from .speculative import SpeculativePipeline, DS41_MODE
    from .generation import GreedyEngine
    from .worker import StageClient
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.94, 0)
    config = Config.from_directory(args.model)
    tokenizer = Tokenizer.from_config(config)
    stage = load_stage(args.model, args.weights_manifest, 0, 20, context=args.context,
                       sessions=args.sessions, depth=args.depth, max_rows=args.max_rows or args.chunk)
    class Local:
        manifest = stage.manifest
        def call(self, op, args): return stage.execute(op, args)
        def close(self): stage.close()
    deadline = time.monotonic() + 900
    while True:
        try:
            remote = StageClient(args.peer, args.peer_port, token, timeout=900)
            break
        except ConnectionRefusedError:
            if time.monotonic() >= deadline: raise
            time.sleep(1)
    pipe = SpeculativePipeline([Local(), remote], mode=DS41_MODE)
    if args.diagnose:
        from .ds41_diagnose import diagnose
        diagnose(pipe,tokenizer,args.diagnose)
    if args.qualify:
        from .ds41_qualify import qualify
        qualify(pipe, tokenizer, args.qualify)
    elif args.draft_depth is not None:
        if args.draft_depth > pipe.depth: raise ValueError('draft depth exceeds worker capacity')
        pipe.depth = args.draft_depth
    if args.optimize:
        from .ds41_optimize import optimize
        result = optimize(pipe,tokenizer,args.optimize)
        args.chunk = result['selected_chunk']
        os.environ['DS41_CLUSTER_ADAPTIVE'] = '1' if result['adaptive_selected'] else '0'
    from .ds41_prefix import PrefixPool
    from .round_policy import RoundPolicy
    mark = lambda text: int(tokenizer.encode(text,add_bos=False,encode_special_tokens=True).flatten()[0])
    pool = PrefixPool(pipe,user_id=mark('<｜User｜>'),assistant_id=mark('<｜Assistant｜>')) if os.environ.get('DS41_CLUSTER_PREFIX','0')=='1' else None
    factory = (lambda:RoundPolicy(pipe.depth)) if os.environ.get('DS41_CLUSTER_ADAPTIVE','0')=='1' else None
    Core = GreedyEngine
    if os.environ.get('DS41_CLUSTER_PREFILL_PIPELINE','0') == '1':
        from .prefill_core import PrefillEngine
        Core = PrefillEngine
    core = Core(pipe, vocab_size=tokenizer.actual_vocab_size, prefill_chunk=args.chunk,
                        adaptive_drafting=False,prefix_pool=pool,policy_factory=factory)
    engine = DeepSeekClusterEngine(core, tokenizer, config.eos_token_id_list)
    print(json.dumps(dict(cluster_ready=True, source=stage.manifest['source_id'],
        context=args.context, sessions=args.sessions, allocated_bytes=torch.cuda.memory_allocated())), flush=True)
    try: web.run_app(create_app(engine, args.model_id), host=args.host, port=args.port)
    finally: pipe.close()


if __name__ == '__main__': main()
