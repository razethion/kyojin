"""Bounded, one-load DeepSeek cluster capacity and greedy DSpark frontier."""
import json
from pathlib import Path
import time
import torch
from .speculative import greedy_prefix


def qualify(pipe, tokenizer, state_dir, count=64):
    state = Path(state_dir); state.mkdir(parents=True,exist_ok=True)
    result = dict(start=time.time(), manifests=[c.manifest for c in pipe.clients], capacity=[], runs=[])
    def save(): (state/'qualification.json').write_text(json.dumps(result,indent=2))
    save()
    for name in ('capacity_a','capacity_b'):
        pipe.open(name)
        result['capacity'].append([c.call('capacity',{'session':name}) for c in pipe.clients])
        save()
    for name in ('capacity_a','capacity_b'): pipe.release(name)
    from tools.ds41 import ds41_chat
    prompts = [ds41_chat.render_chat([dict(role='user', content=s)], thinking=False) for s in (
        'Explain why tides vary through the day in plain language. Give a useful example.',
        'Write a Python function that merges overlapping intervals. Explain its edge cases briefly.')]
    ids = [tokenizer.encode(p,add_bos=False,encode_special_tokens=True).cpu() for p in prompts]
    depth_limit = pipe.depth
    def run(prompt,depth,label,n):
        name='qualify';pipe.open(name); past=0
        pre=time.perf_counter();prefill=prompt[:,:-1]
        for start in range(0,prefill.shape[1],256):
            chunk=prefill[:,start:start+256];pipe.prepare(name,chunk);pipe.commit(name,chunk.shape[1])
        pre=time.perf_counter()-pre;current=prompt[:,-1:];out=[];accepted=proposed=rounds=0
        began=time.perf_counter()
        while len(out)<n:
            position=pipe.sessions.items[name].length
            d=min(depth,pipe.sessions.context-position-1) if position>=2 else 0
            draft=pipe.draft(name,current,d,adaptive=False)['tokens'] if d else torch.empty((1,0),dtype=torch.int64)
            x=torch.cat((current,draft),dim=1)
            predictions=pipe.prepare(name,x,speculative=bool(d))['tokens'][0].tolist()
            decision=greedy_prefix(draft[0].tolist(),predictions,n-len(out))
            pipe.commit(name,decision['committed'])
            accepted+=decision['accepted'];proposed+=draft.numel();rounds+=1
            out+=decision['emitted'];current=torch.tensor([[out[-1]]])
        elapsed=time.perf_counter()-began;pipe.release(name)
        rec=dict(label=label,depth=depth,tokens=out,text=tokenizer.decode(torch.tensor(out),decode_special_tokens=True),
                 seconds=elapsed,tps=n/elapsed,prefill_seconds=pre,accepted=accepted,proposed=proposed,rounds=rounds)
        result['runs'].append(rec);save();print(json.dumps({k:v for k,v in rec.items() if k not in ('tokens','text')}),flush=True)
        return rec
    run(ids[0],0,'warmup',8)
    refs=[run(p,0,f'prompt{i}',count)['tokens'] for i,p in enumerate(ids)]
    candidates=[]
    for depth in range(1,depth_limit+1):
        rows=[]
        for i,p in enumerate(ids):
            rec=run(p,depth,f'prompt{i}',count)
            if rec['tokens']!=refs[i]:
                result['parity_failure']=dict(depth=depth,prompt=i,
                    first_difference=next(k for k,(a,b) in enumerate(zip(rec['tokens'],refs[i])) if a!=b))
                save();raise RuntimeError('DSpark output differs from serial target; qualification stopped')
            rows.append(rec)
        candidates.append(dict(depth=depth,tps=count*len(rows)/sum(r['seconds'] for r in rows),
            accepted=sum(r['accepted'] for r in rows),proposed=sum(r['proposed'] for r in rows)))
    result.update(frontier=candidates,selected_depth=max(candidates,key=lambda r:r['tps'])['depth'],
                  all_tokens_exact=True,completed=time.time())
    pipe.depth=result['selected_depth'];save();return result
