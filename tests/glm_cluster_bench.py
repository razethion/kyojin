"""Bounded opt-in two-node GLM4.10 smoke/speed test; no solo-size claims."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('role', choices=('worker', 'cluster'))
ap.add_argument('--model', required=True)
ap.add_argument('--manifest', required=True)
ap.add_argument('--state', required=True)
ap.add_argument('--token-file', required=True)
ap.add_argument('--depth', type=int, choices=(0,1,2,3), default=0)
ap.add_argument('--variant', choices=('baseline','verify','graphs','k8'), default='baseline')
ap.add_argument('--host', default='127.0.0.1', help='worker bind address or coordinator peer address')
ap.add_argument('--port', type=int, default=9034)
ap.add_argument('--usb4-interface', help='worker only: explicit interface for a direct USB4 bind')
args = ap.parse_args()
os.environ['EXL3_KVWIN'] = '0'
os.environ['EXL3_BLOCK_GRAPH'] = '0'
os.environ['EXL3_VERIFY_FUSE'] = '0' if args.variant=='baseline' else '1'
os.environ['EXL3_VERIFY_FUSE_PAIR'] = '1' if args.variant=='baseline' else '0'
if args.variant=='graphs':
    os.environ.update(EXL3_BLOCK_GRAPH='1',EXL3_BLOCK_GRAPH_MLA='2',EXL3_BLOCK_GRAPH_VERIFY='1')
os.environ['EXL3_DENSE_K8']='1' if args.variant=='k8' else '0'
os.environ['EXL3_GEMV_R_DEC1']='1' if args.variant=='k8' else '0'
for name in ('EXL3_DRAFT2S', 'EXL3_GR_GS_SIDECAR', 'EXL3_MTP_EH_FP16', 'EXL3_MTP_PRENORM_H'):
    os.environ.pop(name, None)
import torch
from exllamav3 import Config, Model, Tokenizer
from tools.multinode.stage import load_stage
from tools.multinode.worker import CachedPipeline, StageClient
from tools.multinode.speculative import SpeculativePipeline, GLM_MODE, greedy_prefix

torch.set_num_threads(4)
torch.set_grad_enabled(False)
torch.cuda.set_per_process_memory_fraction(.94, 0)
config = Config.from_directory(args.model)
assert config.num_hidden_layers == 45 and config.architecture == 'Glm5NextForConditionalGeneration'
shape = Model.from_config(config)
boundary, total = shape.first_block_idx + 24, len(shape.modules)
del shape
state = Path(args.state)
if args.role == 'worker':
    from tools.multinode.worker import main
    sys.argv = ['worker', '--model', args.model, '--weights-manifest', args.manifest,
        '--modules', f'{boundary}:{total}', '--device', 'cuda:0',
        '--host', args.host, '--port', str(args.port),
        '--once', '--accept-timeout', '900', '--token-file', args.token_file,
        '--cache-context', '2048', '--cache-sessions', '2', '--cache-bits', '8',
        '--max-rows', '256', '--experimental-glm', '--speculation-depth', str(args.depth)]
    if args.usb4_interface: sys.argv += ['--usb4-interface', args.usb4_interface]
    main()
    print(json.dumps(dict(worker_peak=torch.cuda.max_memory_allocated())), flush=True)
    raise SystemExit

stage = load_stage(args.model, 0, boundary, 'cuda:0', args.manifest,
    max_rows=256, cache_context=2048, cache_sessions=2, cache_bits=8,
    experimental_glm=True, speculation_depth=args.depth)
print(json.dumps(dict(loaded=True, allocated=torch.cuda.memory_allocated(),
                      manifest=stage.manifest)), flush=True)

class Local:
    manifest = stage.manifest
    def call(self, op, values):
        return stage.execute(op, values)
    def close(self):
        pass

token = bytes.fromhex(Path(args.token_file).read_text().strip())
deadline = time.monotonic() + 600
while True:
    try:
        remote = StageClient(args.host, args.port, token, timeout=900)
        break
    except ConnectionRefusedError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(1)
pipe = SpeculativePipeline([Local(), remote], mode=GLM_MODE) if args.depth else CachedPipeline([Local(), remote])
tokenizer = Tokenizer.from_config(config)
from jinja2.sandbox import ImmutableSandboxedEnvironment
environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
    extensions=['jinja2.ext.loopcontrols'])
template = environment.from_string((Path(args.model) / 'chat_template.jinja').read_text())
prompts = [
    'Write a long, vivid story about a lighthouse keeper discovering a forgotten island. Continue for at least 800 words.',
    'Write a detailed practical guide to planning a community garden, covering its first year. Continue for at least 800 words.',
    'Explain how a small city could improve public transportation over five years, with concrete examples. Continue for at least 800 words.']
report = dict(model='turboderp GLM-5.3-Flash sc_4.10bpw', depth=args.depth, variant=args.variant,greedy_only=True,
    cache_context=2048, cache_sessions=2, boundary=boundary, total_modules=total,
    source_id=stage.manifest['source_id'], model_id=stage.manifest['model_id'],
    engine_env={k:os.environ.get(k) for k in ('EXL3_VERIFY_FUSE','EXL3_VERIFY_FUSE_PAIR','EXL3_BLOCK_GRAPH','EXL3_BLOCK_GRAPH_MLA','EXL3_BLOCK_GRAPH_VERIFY','EXL3_DENSE_K8','EXL3_GEMV_R_DEC1')},
    thinking='template open-think immediately closed for generation measurement', runs=[])
if args.variant=='k8':
    report['serial_reference_note']='R1 target includes unused draft catch-up; parity oracle, not pure target-only speed'

def save():
    prefix='cluster' if args.variant=='baseline' else f'cluster-{args.variant}'
    (state / f'{prefix}-depth{args.depth}-progress.json').write_text(json.dumps(report, indent=2)+'\n')

try:
    plan=[(0,16,'warmup')]+[(i,96,'mtp') for i in range(3)]
    if args.variant=='k8':
        plan=[(0,16,f'warmup{d}') for d in range(1,args.depth+1)]+[
            (i,96,mode) for i in range(3) for mode in ['serial-reference']+[
                f'mtp{d}' for d in range(1,args.depth+1)]]
    references={}
    for index, limit,run_mode in plan:
        active_depth=int(run_mode[-1]) if run_mode.startswith(('mtp','warmup')) and run_mode[-1].isdigit() else args.depth
        name = 'run'
        text = template.render(messages=[dict(role='user', content=prompts[index])],
                               add_generation_prompt=True, tools=None)
        assert text.endswith('<think>')
        text += '\n</think>\n\n'
        ids = tokenizer.encode(text, add_bos=False, encode_special_tokens=True)
        pipe.open(name)
        begin = time.perf_counter()
        if args.depth:
            result = pipe.prepare(name, ids)
            pipe.commit(name, ids.shape[1])
            current = result['tokens'][:, -1:].cpu()
        else:
            result = pipe.forward(name, ids)
            current = result[:, -1:, :config.vocab_size].argmax(-1).cpu()
        torch.cuda.synchronize()
        first = time.perf_counter()
        emitted = current.flatten().tolist()
        rounds = []
        while len(emitted) < limit:
            remaining = limit-len(emitted)
            if args.depth and run_mode!='serial-reference':
                started=time.perf_counter()
                proposals = pipe.draft(name, current, active_depth, adaptive=False)
                drafted=time.perf_counter()
                drafts = proposals['tokens']
                verified = pipe.prepare(name, torch.cat((current, drafts), dim=1), speculative=True)
                checked=time.perf_counter()
                decision = greedy_prefix(drafts.flatten().tolist(), verified['tokens'].flatten().tolist(), remaining)
                pipe.commit(name, decision['committed'])
                committed=time.perf_counter()
                emitted.extend(decision['emitted'])
                current = torch.tensor([[decision['emitted'][-1]]], dtype=torch.int64)
                rounds.append(dict(drafts=drafts.flatten().tolist(), accepted=decision['accepted'],
                                   committed=decision['committed'],draft_ms=1000*(drafted-started),
                                   verify_ms=1000*(checked-drafted),commit_ms=1000*(committed-checked)))
            elif args.depth:
                result=pipe.prepare(name,current)
                pipe.commit(name,1)
                current=result['tokens'][:,-1:].cpu()
                emitted.extend(current.flatten().tolist())
            else:
                result = pipe.forward(name, current)
                current = result[:, -1:, :config.vocab_size].argmax(-1).cpu()
                emitted.extend(current.flatten().tolist())
        torch.cuda.synchronize()
        end = time.perf_counter()
        pipe.release(name)
        record = dict(prompt=index, mode=run_mode,draft_depth=active_depth,warmup=limit==16, tokens=emitted,
            text=tokenizer.decode(torch.tensor([emitted], dtype=torch.int64))[0],
            ttft_ms=1000*(first-begin), tps=(len(emitted)-1)/(end-first),
            accepted_drafts=sum(r['accepted'] for r in rounds),
            proposed_drafts=sum(len(r['drafts']) for r in rounds), rounds=rounds)
        report['runs'].append(record)
        save()
        if run_mode=='serial-reference':references[index]=emitted
        elif args.variant=='k8' and run_mode.startswith('mtp'):
            assert emitted==references[index],f'optimized MTP differs from serial R1 at prompt{index}'
        print(json.dumps({key:value for key,value in record.items() if key not in ('tokens','rounds')}), flush=True)
    report['peak_allocated'] = torch.cuda.max_memory_allocated()
    report['complete'] = True
    save()
except BaseException as error:
    report['failure'] = dict(type=type(error).__name__, message=str(error))
    save()
    raise
finally:
    pipe.close()
    stage.model.unload()
