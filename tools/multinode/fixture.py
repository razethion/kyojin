"""Small deterministic FP16 checkpoint for native cluster correctness gates."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from .identity import manifest


def create(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    assert not list(root.iterdir()),'Fixture destination must be empty'
    torch.manual_seed(713)
    hidden,intermediate,head_dim=256,512,128
    config=dict(architectures=['LlamaForCausalLM'],hidden_size=hidden,intermediate_size=intermediate,
                num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=head_dim,
                vocab_size=256,rms_norm_eps=1e-5,hidden_act='silu',max_position_embeddings=512,
                rope_theta=10000.0,tie_word_embeddings=False,torch_dtype='float16')
    (root/'config.json').write_text(json.dumps(config))
    weights={}
    def random(name,*shape):weights[name+'.weight']=(torch.randn(shape)*.05).half()
    random('model.embed_tokens',256,hidden)
    random('lm_head',256,hidden)
    weights['model.norm.weight']=torch.ones(hidden).half()
    for i in range(2):
        prefix=f'model.layers.{i}'
        for norm in ('input_layernorm','post_attention_layernorm'):
            weights[f'{prefix}.{norm}.weight']=torch.ones(hidden).half()
        for name,rows in (('q_proj',hidden),('k_proj',head_dim),('v_proj',head_dim),('o_proj',hidden)):
            random(f'{prefix}.self_attn.{name}',rows,hidden)
        for name,rows,cols in (('gate_proj',intermediate,hidden),('up_proj',intermediate,hidden),('down_proj',hidden,intermediate)):
            random(f'{prefix}.mlp.{name}',rows,cols)
    save_file(weights,str(root/'model.safetensors'))
    (root/'integrity.json').write_text(json.dumps(manifest(root),indent=2))
