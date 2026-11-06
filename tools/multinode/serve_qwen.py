"""Experimental loopback Qwen cluster server using Kyojin's existing HTTP API.

Start the matching final-stage worker first. This entry point never edits the
primary service, OpenClaw configuration or model weights. Greedy text MTP only.
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def load_http():
    # Import shared helpers directly: Windows source archives can contain Git
    # symlink target text instead of usable tools/qwen/*.py symlinks.
    for name in ('image_cache', 'serve_metrics', 'serve_ping', 'serve_reqlog'):
        module = importlib.import_module('tools.strix_halo.' + name)
        sys.modules.setdefault(name, module)
    return importlib.import_module('tools.qwen.serve')


def create_app(engine, model_id, template):
    from aiohttp import web
    http = load_http()
    engine.bad_request = http.BadRequest
    defaults = dict(http.load_defaults('/nonexistent'), temperature=0, top_p=1,
                    top_k=0, min_p=0, enable_thinking=False)
    app = http.create_app(engine, model_id, template, defaults=defaults, default_max_tokens=128)

    @web.middleware
    async def capabilities(request, handler):
        if engine.core.closed or engine.core.pipeline.failed:
            return web.json_response({'error': {'message': 'Cluster workers require restart',
                                      'type': 'unavailable_error', 'code': 503}}, status=503)
        if request.method == 'POST' and request.path in ('/v1/chat/completions', '/v1/completions', '/completion'):
            try:
                body = await request.json()
            except (ValueError, TypeError):
                return await handler(request)
            if isinstance(body, dict):
                unsupported = [key for key in ('thinking_budget', 'logprobs', 'top_logprobs')
                               if body.get(key) is not None and body.get(key) is not False]
                unsupported += [key for key in ('loop_guard', 'loop_final') if body.get(key)]
                temperature, top_k = body.get('temperature'), body.get('top_k')
                if temperature not in (None, 0) and top_k != 1:
                    unsupported.append('sampled decoding')
                if body.get('speculative') is False:
                    unsupported.append('plain decoding')
                if unsupported:
                    return web.json_response({'error': {'message': 'Experimental cluster does not support: ' + ', '.join(unsupported),
                                              'type': 'invalid_request_error', 'code': 400}}, status=400)
        return await handler(request)

    app.middlewares.insert(0, capabilities)
    async def close(_):
        # Existing HTTP cleanup cancels its request drivers; await them before
        # releasing the distributed channel and unloading the local stage.
        import asyncio
        await asyncio.gather(*app['worker_tasks'], return_exceptions=True)
        await engine.close()
    app.on_cleanup.append(close)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'weights-manifest', 'env', 'token-file'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--peer', default='10.203.0.2')
    parser.add_argument('--peer-port', type=int, default=9032)
    parser.add_argument('--modules', default='0:27')
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--sessions', type=int, default=2)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--port', type=int, default=8742)
    parser.add_argument('--model-id', default='qwen-6.05-cluster-experimental')
    args = parser.parse_args()
    if not 8 <= args.chunk <= 2048 or not 1 <= args.port <= 65535 or not 1 <= args.peer_port <= 65535:
        parser.error('chunk must be8..2048 and ports1..65535')
    from .session import Sessions
    Sessions(args.context, args.sessions)  # validate fixed-page geometry before loading weights
    for key, value in json.loads(Path(args.env).read_text()).items():
        if key not in ('EXL3_ROOT', 'EXL3_VENV', 'EXL3_ROCM_SDK'):
            os.environ[key] = value
    os.environ['EXL3_KVWIN'] = '0'
    http = load_http()
    print(http.configure_hc_sidecar(args.model), flush=True)
    print(http.configure_draft2s(args.model), flush=True)
    token_path = Path(args.token_file)
    stat = token_path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        parser.error('worker token must be private and user-owned')
    token = bytes.fromhex(token_path.read_text().strip())
    if len(token) < 32:
        parser.error('worker token must have at least 32 random bytes')
    from aiohttp import web
    import torch
    from exllamav3 import Config, Tokenizer
    from .generation import GreedyEngine
    from .qwen_engine import QwenClusterEngine
    from .speculative import SpeculativePipeline
    from .stage import load_stage
    from .worker import StageClient
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    config = Config.from_directory(args.model)
    tokenizer = Tokenizer.from_config(config)
    begin, end = map(int, args.modules.split(':'))
    if begin != 0:
        parser.error('HTTP coordinator must own the first stage')
    stage = load_stage(args.model, begin, end, 'cuda:0', args.weights_manifest,
                       max_rows=args.chunk, cache_context=args.context, cache_sessions=args.sessions,
                       cache_bits=8, experimental_qwen=True, speculation_depth=7)
    class Local:
        manifest = stage.manifest
        def call(self, op, values): return stage.execute(op, values)
        def close(self): pass
    pipeline = None
    try:
        remote = StageClient(args.peer, args.peer_port, token, timeout=120)
        pipeline = SpeculativePipeline([Local(), remote])
        core = GreedyEngine(pipeline, vocab_size=tokenizer.actual_vocab_size, prefill_chunk=args.chunk)
        engine = QwenClusterEngine(core, tokenizer, config.eos_token_id_list)
        template = http.DEFAULT_TEMPLATE.read_text()
        app = create_app(engine, args.model_id, template)
        print(json.dumps(dict(cluster_ready=True, greedy_only=True, context=args.context,
                              sessions=args.sessions, source=stage.manifest['source_id'])), flush=True)
        web.run_app(app, host='127.0.0.1', port=args.port)
    finally:
        if pipeline: pipeline.close()
        stage.model.unload()


if __name__ == '__main__':
    main()
