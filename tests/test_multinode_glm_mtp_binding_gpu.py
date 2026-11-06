"""Real GLM MTP native-target versus final-stage binding; target not loaded."""
import argparse
import json
import os
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--model',required=True)
ap.add_argument('--out',required=True)
args=ap.parse_args()
for name in ('EXL3_DRAFT2S','EXL3_MTP_PRENORM_H','EXL3_MTP_EH_FP16'):
    os.environ.pop(name,None)
os.environ['EXL3_BLOCK_GRAPH']='0'
import torch
from exllamav3 import Config,Model,Cache
from tools.multinode.speculative import TargetBinding
torch.set_grad_enabled(False);torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(753)
config=Config.from_directory(args.model)
target=Model.from_config(config)
embed,head=target.modules[0],target.modules[-1]
embed.load(torch.device('cpu'));head.load(torch.device('cuda:0'))
draft=Model.from_config(config,component='mtp')
cache=Cache(draft,max_num_tokens=1024,max_batch_size=2)
draft.load(device='cuda:0',progressbar=False)
binding=TargetBinding();binding.config=config
binding.modules=[embed]+target.modules[26:]
binding.logit_layer_idx=len(binding.modules)-1
assert binding.modules[-1] is head and not binding.loaded_tp
operations=[];past=0
for rows in (8,1,1,2):
    operations.append((past,torch.randint(200,400,(1,rows)),
                       torch.randn(1,rows,config.hidden_size,dtype=torch.float16,device='cuda:0')))
    past+=rows
expected=[];records=[]
try:
    for slot,owner in enumerate((target,binding)):
        draft.attach_to(owner)
        tap=sorted(draft.draft_verifier_params['export_state_norm_keys'])
        for index,(past,ids,hidden) in enumerate(operations):
            params=dict(attn_mode='flash_attn',cache=cache,target_hidden=hidden,
                block_table=torch.arange(slot*2,(slot+1)*2,dtype=torch.int32).view(1,2),
                cache_seqlens=torch.tensor([past],dtype=torch.int32),export_draft_logits=True)
            out=draft.forward(ids,params)
            tokens=draft.sample_from_state(out,params)
            assert torch.isfinite(out).all() and torch.isfinite(params['draft_logits']).all()
            actual=(out.cpu().clone(),tokens.cpu().clone(),params['draft_logits'].cpu().clone())
            if slot==0:expected.append(actual)
            else:
                for a,b in zip(actual,expected[index]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                records.append(dict(rows=ids.shape[1],past=past,exact=True,norm_tap=tap))
                print(json.dumps(records[-1]),flush=True)
    Path(args.out).write_text(json.dumps(dict(passed=True,cases=records,
        peak_allocated=torch.cuda.max_memory_allocated()),indent=2)+'\n')
finally:
    draft.unload();head.unload();embed.unload()
