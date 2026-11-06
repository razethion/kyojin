"""Tiny GPU-output/CPU-embedding draft-loop regression; no checkpoint load."""
import torch
from tools.multinode.speculative import SpeculativeQwenStage
from tools.multinode.session import Sessions
torch.set_num_threads(2)
stage=object.__new__(SpeculativeQwenStage)
stage.final=True;stage.depth=7;stage.sessions=Sessions(512,1);stage.sessions.open('a')
stage.sessions.items['a'].length=17;stage.draft_pending={}
stage.carry={'a':torch.ones((1,1,8),device='cuda:0')};stage.vocab=10;stage.draft_cache=None
seen=[]
class Draft:
    def forward(self,ids,params):
        assert ids.device.type=='cpu','shared CPU embedding received GPU token IDs'
        seen.append(ids.clone());return torch.ones((1,1,8),device='cuda:0')
    def sample_from_state(self,hidden,params):
        params['draft_p']=torch.tensor([.99],device='cuda:0')
        return torch.tensor([[3]],device='cuda:0')
stage.draft_model=Draft()
with torch.inference_mode():
    result=stage.draft(dict(session='a',step=0,past=17,input_ids=torch.tensor([[2]]),depth=3,adaptive=False))
assert len(seen)==3 and result['tokens'].device.type=='cpu'
assert result['tokens'].tolist()==[[3,3,3]] and [x.item() for x in seen]==[2,3,3]
print('PASS: all3 MTP forwards receive CPU IDs after GPU head sampling; peakbytes='+str(torch.cuda.max_memory_allocated()))
