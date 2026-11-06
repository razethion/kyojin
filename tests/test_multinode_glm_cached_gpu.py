"""Native GLM5Next versus cached stages: KDA, mHC, sparse MLA, page/slot reuse."""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
os.environ['EXL3_KVWIN'] = '0'
os.environ['EXL3_BLOCK_GRAPH'] = '0'
import torch
from exllamav3 import Cache, CacheLayer_quant, Config, Model
from tools.multinode.stage import CachedGlmStage, source_fingerprint
from tools.multinode.wire import pack, unpack

torch.set_num_threads(4)
torch.set_grad_enabled(False)
torch.cuda.set_per_process_memory_fraction(.85, 0)
torch.manual_seed(41102)
records = []
with tempfile.TemporaryDirectory(prefix='cluster-glm-fixture-') as temporary:
    root = Path(temporary)
    for path in Path(args.model).iterdir():
        if path.is_file() and path.suffix in ('.json', '.jinja', '.safetensors'):
            shutil.copy2(path, root / path.name)
    config_data = json.loads((root / 'config.json').read_text())
    text = config_data['text_config']
    text['index_topk'] = 32  # exercise real sparse index selection within512tokens
    assert all(t == 'full' for t in text['indexer_types'])
    (root / 'config.json').write_text(json.dumps(config_data))
    config = Config.from_directory(str(root))
    count = config.num_hidden_layers
    assert count >= 4 and any(t == 'deepseek_sparse_attention' for t in config.layer_types[:count])
    operations = []
    lengths, steps = {'a': 0, 'b': 0}, {'a': 0, 'b': 0}
    # This public fixture has64-wide KDA heads: native recurrent decode only
    # supports128. Keep this gate on the native chunk path; actual-model
    # slice gates must separately qualify single-token decode and rollback.
    for name, rows in [('a', 250), ('b', 17), ('a', 12), ('b', 4), ('a', 4), ('b', 4)]:
        ids = torch.randint(10, min(512, config.vocab_size), (1, rows), dtype=torch.int64)
        operations.append((name, steps[name], lengths[name], ids))
        lengths[name] += rows; steps[name] += 1
    operations.append(('c', 0, 0, torch.randint(10, min(512, config.vocab_size), (1, 7))))
    for bits in (16, 8):
        model = Model.from_config(config)
        cache = Cache(model, max_num_tokens=1024, max_batch_size=2, max_history=2,
                      **(dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8) if bits == 8 else {}))
        model.load(device='cuda:0', progressbar=False)
        if os.environ.get('GLM_DSA_TRACE') == '1':
            from safetensors.torch import save_file
            for block in model.modules:
                attention = getattr(block, 'attn', None)
                if attention is None or getattr(attention, 'indexer_mode', None) != 'full': continue
                sparse_forward, post_forward = attention._attend_sparse, attention.attend_post
                def sparse_trace(*values, _fn=sparse_forward, _scale=attention.sm_scale, **kw):
                    q, qp, batch, rows, params, pool, rope, table, indices, qc, *tail = values
                    past = int(params['cache_seqlens'][0])
                    bad = (indices >= torch.arange(past+1, past+rows+1, device=indices.device)[:, None]) & (indices >= 0)
                    print(json.dumps(dict(dsa_inputs=True, query_finite=bool(torch.isfinite(q).all()),
                        query_shape=list(q.shape), pool_shape=list(pool.shape), causal_violations=int(bad.sum()),
                        first_indices=indices[:4].cpu().tolist())), flush=True)
                    save_file(dict(q=q.contiguous().cpu(), pool=pool.contiguous().cpu(), table=table.contiguous().cpu(),
                                   indices=indices.contiguous().cpu()), str(Path(args.out).with_suffix('.dsa-inputs.st')),
                              metadata={'scale': str(_scale), 'past': str(past)})
                    return _fn(*values, **kw)
                def post_trace(latent, *values, _fn=post_forward, **kw):
                    print(json.dumps(dict(latent_finite=bool(torch.isfinite(latent).all()),
                                          nonfinite_elements=int((~torch.isfinite(latent)).sum()))), flush=True)
                    return _fn(latent, *values, **kw)
                attention._attend_sparse, attention.attend_post = sparse_trace, post_trace
        states, expected = {}, []
        with torch.inference_mode():
            states = {name: cache.get_new_state() for name in ('a', 'b')}
            diagnostic = []
            for module in (part for top in model.modules for part in top):
                original = module.forward
                def traced(*a, _fn=original, _key=module.key, **kw):
                    value = _fn(*a, **kw)
                    if isinstance(value, torch.Tensor):
                        finite = bool(torch.isfinite(value).all())
                        diagnostic.append(dict(module=_key, shape=list(value.shape), finite=finite))
                        if not finite:
                            print(json.dumps({'native_nonfinite': diagnostic[-20:]}), flush=True)
                            raise FloatingPointError('first native nonfinite at ' + _key)
                    return value
                module.forward = traced
            for name, step, past, ids in operations:
                if name == 'c': states.pop('a').free(); states['c'] = cache.get_new_state()
                slot = 1 if name == 'b' else 0
                params = dict(attn_mode='flash_attn', cache=cache, last_tokens_only=1,
                    block_table=torch.arange(slot*2, (slot+1)*2, dtype=torch.int32).view(1, 2),
                    cache_seqlens=torch.tensor([past], dtype=torch.int32), recurrent_states=[states[name]])
                out = model.forward(ids, params).cpu()
                assert torch.isfinite(out).all()
                assert states[name].position == past + ids.shape[1]
                expected.append(out)
            for state in states.values(): state.free()
        model.unload()
        total = len(model.modules)
        # Split directly before the sparse layer and after it, covering both
        # possible placements of its indexer/cache relative to the USB boundary.
        sparse = next(i for i, kind in enumerate(config.layer_types[:count]) if kind == 'deepseek_sparse_attention')
        for boundary in sorted({2 + sparse, 3 + sparse}):
            if boundary >= 2 + count: continue
            stages = [CachedGlmStage(Model.from_config(config), begin, end, device='cuda:0',
                       model_id='tiny-glm', source_id='same', max_rows=256,
                       context=512, sessions=2, cache_bits=bits)
                      for begin, end in ((0, boundary), (boundary, total))]
            try:
                for stage in stages:
                    stage.load()
                    stage.execute('open', {'session': 'a'}); stage.execute('open', {'session': 'b'})
                owned = [set(stage.cache.layers) for stage in stages]
                assert not owned[0] & owned[1]
                for index, (name, step, past, ids) in enumerate(operations):
                    if name == 'c':
                        for stage in stages:
                            stage.execute('release', {'session': 'a'}); stage.execute('open', {'session': 'c'})
                    activation = None
                    for stage in stages:
                        request = dict(session=name, step=step, past=past, input_ids=ids,
                                       activation=activation, last_tokens_only=1)
                        header, payload = pack(request)
                        activation = stage.execute('prepare', unpack(header, payload))
                    torch.testing.assert_close(activation.cpu(), expected[index], rtol=0, atol=0,
                                               msg=f'bits{bits} split{boundary} {name}:{step}')
                    for stage in stages:
                        assert stage.execute('commit', dict(session=name, step=step)) == dict(length=past+ids.shape[1], step=step+1)
                record = dict(cache_bits=bits, boundary=boundary, exact_steps=len(operations),
                              sparse_indexer=True, page_crossing=True, slot_reuse=True)
                records.append(record); print(json.dumps(record), flush=True)
            finally:
                for stage in stages: stage.model.unload()
    result = dict(passed=True, records=records, peak_allocated=torch.cuda.max_memory_allocated(),
                  source_id=source_fingerprint(Path(__file__).resolve().parents[1]), fixture_index_topk=32)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
