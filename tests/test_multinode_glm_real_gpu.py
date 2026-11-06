"""Unchanged GLM4.10 first-eight-block adapter/native decode and rewind gate.

No full-model solo comparison is claimed: the model exceeds one node's RAM.
This qualifies real KDA128, MLA512, mHC and native kpool prefix semantics.
"""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
os.environ['EXL3_KVWIN'] = '0'
os.environ['EXL3_BLOCK_GRAPH'] = '0'
import torch
from exllamav3 import Config, Model
from tools.multinode.speculative import SpeculativeGlmStage
from tools.multinode.stage import source_fingerprint

torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.90, 0)
torch.manual_seed(513021)
config = Config.from_directory(args.model)
assert config.architecture == 'Glm5NextForConditionalGeneration'
assert config.num_hidden_layers == 45 and config.kv_lora_rank == 512
context, pages, width = 2560, 10, 3
model = Model.from_config(config)
stage = SpeculativeGlmStage(model, 0, model.first_block_idx + 8,
    speculation_depth=2, device='cuda:0', model_id='verified-public-sc4.10',
    source_id=source_fingerprint(Path(__file__).resolve().parents[1]),
    max_rows=256, context=context, sessions=2, cache_bits=8)
stage.load()
print(json.dumps(dict(loaded=True, module_keys=stage.manifest['module_keys'],
                      allocated=torch.cuda.memory_allocated())), flush=True)
records = []
try:
    for prefix in (17, 250, 2050, 2051, 2052):
        for accepted in (1, 2, 3):
            name = 'case'
            stage.execute('open', dict(session=name))
            with torch.inference_mode():
                reference = stage.cache.get_new_state()
            assert stage.states[name].slot != reference.slot

            def native(ids, past, history):
                params = dict(attn_mode='flash_attn', cache=stage.cache,
                    last_tokens_only=ids.shape[1], recurrent_states=[reference],
                    block_table=torch.arange(pages, 2*pages, dtype=torch.int32).view(1, pages),
                    cache_seqlens=torch.tensor([past], dtype=torch.int32),
                    recurrent_history=history, dflash_verify=history)
                with torch.inference_mode():
                    output = stage.model.forward(ids, params)
                    pending = params.pop('hc_pending', None)
                    if pending is not None:
                        stage.flush_pending(output, pending)
                assert torch.isfinite(output).all()
                return output.cpu()

            def prepared(ids, past, step, history):
                output = stage.execute('prepare', dict(session=name, step=step, past=past,
                    input_ids=ids, activation=None, last_tokens_only=ids.shape[1],
                    speculative=history, return_logits=False))
                assert torch.isfinite(output).all()
                return output.cpu()

            prompt = torch.randint(200, 500, (1, prefix))
            step = 0
            for past in range(0, prefix, 256):
                ids = prompt[:, past:past+256]
                torch.testing.assert_close(prepared(ids, past, step, False), native(ids, past, False), rtol=0, atol=0)
                stage.execute('commit', dict(session=name, step=step, accepted=ids.shape[1]))
                step += 1
            window = torch.randint(200, 500, (1, width))
            torch.testing.assert_close(prepared(window, prefix, step, True), native(window, prefix, True), rtol=0, atol=0)
            stage.execute('commit', dict(session=name, step=step, accepted=accepted))
            with torch.inference_mode():
                reference.rewind(width-accepted)
            assert stage.states[name].position == reference.position == prefix + accepted
            assert stage.states[name].last_history == reference.last_history == 0
            follow = torch.tensor([[299]])
            torch.testing.assert_close(prepared(follow, prefix+accepted, step+1, False),
                                       native(follow, prefix+accepted, False), rtol=0, atol=0)
            stage.execute('commit', dict(session=name, step=step+1, accepted=1))
            # Compare every complete pool in the two physical cache page spans.
            pooled = 0
            for layer in stage.cache.layers.values():
                pool = getattr(layer, 'k_pool', None)
                if pool is None:
                    continue
                n = (prefix + accepted + 1)//layer.kpool
                first = pool[:pages].flatten(0, 1)[:n]
                second = pool[pages:2*pages].flatten(0, 1)[:n]
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                pooled += 1
            assert pooled == 2
            stage.execute('release', dict(session=name))
            reference.free()
            record = dict(prefix=prefix, accepted=accepted, rejected=width-accepted,
                          exact=True, complete_pool_layers=pooled)
            records.append(record)
            print(json.dumps(record), flush=True)
            Path(args.out).write_text(json.dumps(dict(passed=False, cases=records), indent=2)+'\n')
    Path(args.out).write_text(json.dumps(dict(passed=True, cases=records,
        source_id=stage.manifest['source_id'], peak_allocated=torch.cuda.max_memory_allocated()), indent=2)+'\n')
finally:
    stage.model.unload()
