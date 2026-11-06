"""One-load teacher-forced R1/R2/cache/drafter isolation on the actual model."""
import json
from pathlib import Path
import time
import torch


def diagnose(pipe,tokenizer,state_dir):
    from tools.ds41 import ds41_chat
    state=Path(state_dir)
    old=json.loads((state/'attempt1-qualification.json').read_text())
    assert old['manifests'][0]['model_id']==pipe.clients[0].manifest['model_id']
    texts=['Explain why tides vary through the day in plain language. Give a useful example.',
           'Write a Python function that merges overlapping intervals. Explain its edge cases briefly.']
    results=dict(start=time.time(),manifests=[c.manifest for c in pipe.clients],arms=[])
    def save(): (state/'diagnostic.json').write_text(json.dumps(results,indent=2))
    def compare(rows,traces):
        return [c.call('debug',dict(action='compare',left='oracle',right='block',rows=rows,traces=traces)) for c in pipe.clients]
    for engram in ('1',):
        for c in pipe.clients:c.call('debug',dict(action='settings',engram=engram))
        for mode in ('full','reject'):
            for pi,text in enumerate(texts):
                reference=next(r['tokens'] for r in old['runs'] if r['label']==f'prompt{pi}' and r['depth']==0)
                prompt=tokenizer.encode(ds41_chat.render_chat([dict(role='user',content=text)],thinking=False),
                                        add_bos=False,encode_special_tokens=True).cpu()
                teachers=torch.tensor([[int(prompt[0,-1])]+reference],dtype=torch.int64)
                arm=dict(engram=engram,mode=mode,prompt=pi,rounds=[]);results['arms'].append(arm)
                for name in ('oracle','block'):
                    pipe.open(name)
                    for pos in range(0,prompt.shape[1]-1,256):
                        x=prompt[:,pos:min(pos+256,prompt.shape[1]-1)]
                        pipe.prepare(name,x);pipe.commit(name,x.shape[1])
                arm['prefill_cache']=compare(1,False)
                for c in pipe.clients:
                    for name in ('oracle','block'):c.call('debug',dict(action='trace',session=name))
                offset=0
                while offset<min(62,len(reference)-1):
                    current=teachers[:,offset:offset+1]
                    keep=2 if mode=='full' else 1
                    if mode=='draft':
                        draft=pipe.draft('block',current,1,adaptive=False)['tokens']
                        x=torch.cat((current,draft),1)
                    else:
                        second=teachers[:,offset+1:offset+2].clone()
                        if mode=='reject':second=(second+17)%tokenizer.actual_vocab_size
                        x=torch.cat((current,second),1)
                    before=compare(1,False) if mode=='draft' else None
                    serial=[]
                    for j in range(keep):
                        y=pipe.prepare('oracle',teachers[:,offset+j:offset+j+1],return_logits=True)
                        pipe.commit('oracle',1);serial.append(y['logits'])
                    expected=torch.cat(serial,dim=1)
                    actual=pipe.prepare('block',x,speculative=True,return_logits=True)['logits'][:,:keep]
                    pipe.commit('block',keep)
                    delta=(actual-expected).abs()
                    rec=dict(offset=offset,rows=2,kept=keep,logits_exact=torch.equal(actual,expected),
                        max_abs=float(delta.max()),argmax_equal=torch.equal(actual.argmax(-1),expected.argmax(-1)),
                        serial_reference=expected.argmax(-1)[0].tolist()==reference[offset:offset+keep],
                        after_draft_cache=before,stages=compare(keep,True))
                    arm['rounds'].append(rec);save()
                    bad=not rec['logits_exact'] or any(s['cache_differences'] for s in rec['stages'])
                    if bad:
                        print(json.dumps(dict(engram=engram,mode=mode,prompt=pi,first=rec)),flush=True);break
                    offset+=keep
                for name in ('oracle','block'):pipe.release(name)
                arm['all_exact']=all(r['logits_exact'] and not any(s['cache_differences'] for s in r['stages']) for r in arm['rounds'])
                save()
    results['completed']=time.time();save()
    raise RuntimeError('Diagnostic complete; inspect first differing operations before promotion')
