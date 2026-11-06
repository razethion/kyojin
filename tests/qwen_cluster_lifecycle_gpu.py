"""Bounded full-checkpoint request lifecycle gate; no serving endpoint changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('role', choices=('solo', 'cluster', 'worker', 'http'))
for key in ('model', 'manifest', 'env', 'state'):
    parser.add_argument('--' + key, required=True)
parser.add_argument('--token-file')
args = parser.parse_args()
for key, value in json.loads(Path(args.env).read_text()).items():
    if key not in ('EXL3_ROOT', 'EXL3_VENV', 'EXL3_ROCM_SDK'):
        os.environ[key] = value
os.environ['EXL3_KVWIN'] = '0'
sidecar = Path(args.model) / 'hc_gs_sidecar.safetensors'
assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == '473e7547adaad190788634556ce6b32614bdb280ccedadbed8789d39b59dcc8f'
os.environ['EXL3_GR_GS_SIDECAR'] = str(sidecar)
assert os.environ.get('EXL3_DRAFT2S') == str(Path(args.model) / 'draft2s.st')

import torch
from exllamav3 import Config, Tokenizer
from tools.multinode.generation import GreedyEngine
from tools.multinode.speculative import SpeculativePipeline
from tools.multinode.stage import load_stage
from tools.multinode.worker import StageClient

torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.94, 0)
state = Path(args.state)
if args.role == 'worker':
    import sys
    from tools.multinode.worker import main
    sys.argv = ['worker', '--model', args.model, '--weights-manifest', args.manifest,
                '--modules', '27:53', '--device', 'cuda:0', '--host', '10.203.0.2',
                '--port', '9032', '--usb4-interface', 'thunderbolt0', '--once',
                '--accept-timeout', '900', '--token-file', args.token_file,
                '--cache-context', '2048', '--cache-sessions', '2', '--cache-bits', '8',
                '--max-rows', '128', '--experimental-qwen', '--speculation-depth', '7']
    main()
    print(json.dumps({'worker_peak': torch.cuda.max_memory_allocated()}), flush=True)
    raise SystemExit

tokenizer = Tokenizer.from_config(Config.from_directory(args.model))
prompts = [
    'Write a long, vivid story about a lighthouse keeper discovering a forgotten island. Continue for at least 800 words.',
    'Write a detailed practical guide to planning a community garden, covering its first year. Continue for at least 800 words.']
short = [tokenizer.encode('<|im_start|>user\n' + text + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
                          add_bos=False, encode_special_tokens=True) for text in prompts]
assert [ids.shape[1] for ids in short] == [38, 38]
# Deterministic extra prose inside the user message; preserve assistant suffix.
padding = tokenizer.encode('The town has a library, a railway station, a park, and several quiet streets. ', add_bos=False)
def extended(size):
    needed = size - short[0].shape[1]
    repeat = padding.repeat(1, (needed + padding.shape[1] - 1) // padding.shape[1])[:, :needed]
    return torch.cat((short[0][:, :3], repeat, short[0][:, 3:]), dim=1)
inputs = [short[0], short[1], extended(600), extended(2032), short[0][:, -1:]]
budgets = [48, 48, 16, 8, 8]
stage = load_stage(args.model, 0, 53 if args.role == 'solo' else 27, 'cuda:0', args.manifest,
                   max_rows=128, cache_context=2048, cache_sessions=2, cache_bits=8,
                   experimental_qwen=True, speculation_depth=7)
class Local:
    manifest = stage.manifest
    def call(self, op, values): return stage.execute(op, values)
    def close(self): pass
clients = [Local()]
if args.role in ('cluster', 'http'):
    previous = json.loads((state / 'lifecycle-solo-result.json').read_text())
    assert previous['model_id'] == stage.manifest['model_id']
    clients.append(StageClient('10.203.0.2', 9032, bytes.fromhex(Path(args.token_file).read_text().strip()), timeout=120))
pipeline = SpeculativePipeline(clients)
engine = GreedyEngine(pipeline, vocab_size=tokenizer.actual_vocab_size, prefill_chunk=128)
if args.role == 'http':
    import asyncio
    from multinode_http_gpu_gate import run_http_gate
    try:
        result = asyncio.run(run_http_gate(engine, tokenizer, tokenizer.config.eos_token_id_list,
                                          inputs, previous['records']))
        result.update(source_id=stage.manifest['source_id'], model_id=stage.manifest['model_id'],
                      peak_allocated=torch.cuda.max_memory_allocated())
        (state / 'http-cluster-result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        engine.close()
        stage.model.unload()
    raise SystemExit
report = dict(role=args.role, model_id=stage.manifest['model_id'], source_id=stage.manifest['source_id'],
              context=2048, sessions=2, depth=7, prefill_chunk=128, records=[])
def save(suffix='progress'):
    (state / f'lifecycle-{args.role}-{suffix}.json').write_text(json.dumps(report, indent=2) + '\n')
def record(label, request, events):
    tokens = [token for event in events for token in event['token_ids']]
    item = dict(label=label, tokens=tokens, reason=request.finish_reason,
                generated=request.generated, prefill_steps=sum(event['phase'] == 'prefill' for event in events))
    report['records'].append(item)
    save()
    print(json.dumps({**item, 'tokens': len(tokens)}), flush=True)
    return tokens
def run(label, ids, budget, stop=()):
    request = engine.start(ids, max_new_tokens=budget, stop_ids=stop)
    return record(label, request, list(request))
try:
    refs = [run(f'reference-{i}', ids, budget) for i, (ids, budget) in enumerate(zip(inputs, budgets))]
    if args.role == 'solo':
        old = json.loads((state / 'solo-mtp-result.json').read_text())
        for i in range(2):
            prior = next(row for row in old['adapted'] if row['phase'] == 'check' and row['prompt'] == i)
            assert refs[i] == prior['tokens'], 'lifecycle changed prior accepted MTP output'
    # Two simultaneously live native recurrent states and disjoint KV slots.
    requests = [engine.start(inputs[i], max_new_tokens=budgets[i]) for i in (0, 2)]
    events = [[], []]
    while not all(request.done for request in requests):
        for index, request in enumerate(requests):
            if not request.done: events[index].append(request.step())
    for index, ref in enumerate((0, 2)):
        assert record(f'interleaved-{ref}', requests[index], events[index]) == refs[ref]
    # Cancel after a partial prefill and after a complete speculation round.
    for phase in ('prefill', 'generation'):
        request = engine.start(inputs[2 if phase == 'prefill' else 0], max_new_tokens=48)
        events = [request.step()]
        if phase == 'generation':
            while events[-1]['phase'] == 'prefill': events.append(request.step())
        request.cancel(); events.append(request.step())
        record('cancel-' + phase, request, events)
        assert request.finish_reason == 'cancelled'
        assert not pipeline.sessions.items and not stage.states and not stage.carry
        assert run('reuse-' + phase, inputs[0], 48) == refs[0]
    stop_token = refs[0][3]
    stop_index = refs[0].index(stop_token)
    assert run('eos-prefix', inputs[0], 48, [stop_token]) == refs[0][:stop_index]
    assert not pipeline.sessions.items and not stage.states and not stage.carry
    if args.role == 'cluster':
        assert report['records'] == previous['records'], 'cluster lifecycle differs from solo'
    report['peak_allocated'] = torch.cuda.max_memory_allocated()
    report['passed'] = True
    save('result')
finally:
    engine.close()
    if stage.draft_model: stage.draft_model.unload()
    stage.model.unload()
