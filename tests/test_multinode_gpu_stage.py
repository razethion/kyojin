"""Opt-in real EXL3 FP16-kernel span check using tiny random Llama weights.

Not a trained model, quantization test, cached decode test, or two-node proof.
Run only on an otherwise idle test GPU; no checkpoint download required.
"""
import json
import os
import argparse
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

# Current upstream declares KVWIN ring state by default on ROCm, even before
# a cache is attached. This gate intentionally exercises only uncached forward.
os.environ["EXL3_KVWIN"] = "0"

from exllamav3 import Config, Model
from tools.multinode.identity import manifest
from tools.multinode.stage import load_stage
from tools.multinode.wire import pack, unpack


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch',type=int,choices=(1,2),default=1)
    ap.add_argument('--attention',choices=('native','torch'),default='native')
    ap.add_argument('--trace',action='store_true',help='abort at the first nonfinite module output')
    ap.add_argument('--diagnose',action='store_true',help='synchronized per-module parity snapshots for the first mismatching operation')
    args=ap.parse_args()
    batch_size=args.batch
    if args.attention=='torch':
        # Diagnostic only: both full and split models use the same SDPA path.
        import importlib
        dispatch=importlib.import_module('exllamav3.modules.attention_fn.dispatch')
        from exllamav3.modules.attention_fn.torch import fn_torch_sdpa_fallback_nocache
        dispatch.attn_fns[:]=[fn_torch_sdpa_fallback_nocache]
    torch.set_num_threads(2)
    torch.manual_seed(713)
    with tempfile.TemporaryDirectory(prefix="multinode-native-stage-") as temp:
        root = Path(temp)
        hidden,intermediate,head_dim=256,512,128
        config = dict(architectures=["LlamaForCausalLM"], hidden_size=hidden,
                      intermediate_size=intermediate, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=head_dim,
                      vocab_size=256, rms_norm_eps=1e-5, hidden_act="silu",
                      max_position_embeddings=128, rope_theta=10000.0,
                      tie_word_embeddings=False, torch_dtype="float16")
        (root / "config.json").write_text(json.dumps(config))
        weights = {}
        def random(name, *shape):
            weights[name + ".weight"] = (torch.randn(shape) * .05).half()
        random("model.embed_tokens", 256, hidden)
        random("lm_head", 256, hidden)
        weights["model.norm.weight"] = torch.ones(hidden).half()
        for i in range(2):
            prefix = f"model.layers.{i}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                weights[f"{prefix}.{norm}.weight"] = torch.ones(hidden).half()
            for projection, rows in (("q_proj",hidden),("k_proj",head_dim),("v_proj",head_dim),("o_proj",hidden)):
                random(f"{prefix}.self_attn.{projection}", rows,hidden)
            for projection, rows, cols in (("gate_proj",intermediate,hidden),("up_proj",intermediate,hidden),("down_proj",hidden,intermediate)):
                random(f"{prefix}.mlp.{projection}",rows,cols)
        save_file(weights, str(root / "model.safetensors"))
        receipt = root / "integrity.json"
        receipt.write_text(json.dumps(manifest(root)))
        reference = Model.from_config(Config.from_directory(str(root)))
        reference.load(device="cuda:0", progressbar=False)
        traced = {}
        if args.trace or args.diagnose:
            for module in reference:
                forward=module.forward
                key=module.key
                def checked(*call_args,_forward=forward,_key=key,**call_kwargs):
                    output=_forward(*call_args,**call_kwargs)
                    if isinstance(output,torch.Tensor) and not torch.isfinite(output).all():
                        raise RuntimeError(f'first nonfinite module {_key}, shape={tuple(output.shape)}, dtype={output.dtype}')
                    if args.diagnose and isinstance(output,torch.Tensor):
                        traced[_key] = output.cpu().clone()
                    return output
                module.forward=checked
        total = len(reference.modules)
        cases = []
        with torch.inference_mode():
            for batch, rows in ((batch_size,1),(batch_size,7),(batch_size,16)):
                ids = torch.randint(0,256,(batch,rows),dtype=torch.int64)
                for last in sorted({1, rows}):
                    traced.clear()
                    expected = reference.forward(ids, {"last_tokens_only":last}).cpu()
                    assert torch.isfinite(expected).all(), f"nonfinite local reference batch={batch} rows={rows} last={last}"
                    print(f'finite local reference batch={batch} rows={rows} last={last}',flush=True)
                    cases.append((ids,last,expected,dict(traced)))
        reference.unload()
        rows_out=[]
        for split in (1,2,3,4):
            left=load_stage(str(root),0,split,"cuda:0",str(receipt),max_rows=32)
            right=load_stage(str(root),split,total,"cuda:0",str(receipt),max_rows=32)
            if args.trace or args.diagnose:
                for stage in (left,right):
                    for module in stage.model:
                        forward=module.forward
                        key=module.key
                        def checked(*call_args,_forward=forward,_key=key,**call_kwargs):
                            output=_forward(*call_args,**call_kwargs)
                            if isinstance(output,torch.Tensor) and not torch.isfinite(output).all():
                                raise RuntimeError(f'first nonfinite staged module {_key}, shape={tuple(output.shape)}, dtype={output.dtype}')
                            if args.diagnose and isinstance(output,torch.Tensor):
                                traced[_key] = output.cpu().clone()
                            return output
                        module.forward=checked
            for ids,last,expected,expected_trace in cases:
                traced.clear()
                activation=left.forward(dict(input_ids=ids,activation=None,last_tokens_only=last))
                # Exercise the exact byte-preserving CPU boundary used by TCP.
                header,payload=pack(dict(input_ids=ids,activation=activation.cpu(),last_tokens_only=last))
                request=unpack(header,payload)
                actual=right.forward(request).cpu()
                if args.diagnose:
                    for key,value in expected_trace.items():
                        assert key in traced, f'missing traced module {key}'
                        if not torch.equal(traced[key],value):
                            diff=(traced[key].float()-value.float()).abs()
                            print(json.dumps(dict(first_module_mismatch=key,split=split,batch=ids.shape[0],rows=ids.shape[1],last=last,
                                                  shape=list(value.shape),max_abs=float(diff.max()),
                                                  expected_first=value.flatten()[:8].tolist(),actual_first=traced[key].flatten()[:8].tolist())),flush=True)
                            raise AssertionError(f'first module mismatch {key}')
                assert torch.isfinite(actual).all(), f"nonfinite staged result split={split} batch={ids.shape[0]} rows={ids.shape[1]} last={last}, boundary_finite={bool(torch.isfinite(activation).all())}"
                try:
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                except AssertionError as error:
                    raise AssertionError(f'split={split}, batch={ids.shape[0]}, rows={ids.shape[1]}, last={last}: {error}') from error
                rows_out.append(dict(split=split,batch=ids.shape[0],sequence=ids.shape[1],
                                     last=last,activation_dtype=str(activation.dtype),
                                     bitwise_equal=True))
            left.model.unload();right.model.unload()
        print(json.dumps(dict(test="real-exl3-fp16-random-stage",scope="single-GPU uncached",
                              torch=torch.__version__,hip=torch.version.hip,
                              attention=args.attention,
                              synchronized_trace=args.trace,
                              cases=rows_out,total_modules=total),indent=2))


if __name__ == "__main__": main()
