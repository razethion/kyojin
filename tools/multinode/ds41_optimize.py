"""One-load cache correctness, prefill frontier and measured MTP policy gate."""
import json,time,os
from pathlib import Path
import torch
from .generation import GreedyEngine
from .ds41_prefix import PrefixPool
from .round_policy import RoundPolicy


def optimize(pipe,tok,state):
    if os.environ.get('DS41_OPT_FINAL_PREFILL','0')=='1':
        return qualify_prefill(pipe,tok,state)
    from tools.ds41 import ds41_chat
    state=Path(state);result=dict(start=time.time(),runs=[],manifests=[c.manifest for c in pipe.clients])
    def save(): (state/'optimization.json').write_text(json.dumps(result,indent=2))
    mark=lambda t:int(tok.encode(t,add_bos=False,encode_special_tokens=True).flatten()[0])
    pool=PrefixPool(pipe,user_id=mark('<｜User｜>'),assistant_id=mark('<｜Assistant｜>'))
    core=GreedyEngine(pipe,vocab_size=tok.actual_vocab_size,prefill_chunk=256,
                     adaptive_drafting=False,prefix_pool=pool)
    def prompt(messages,thinking=False):
        return tok.encode(ds41_chat.render_chat(messages,thinking=thinking),add_bos=False,encode_special_tokens=True).cpu()
    def run(ids,label,n=24,cache=False,adaptive=False,chunk=256,depth=3):
        pipe.depth=depth;core.prefill_chunk=chunk
        core.policy_factory=(lambda:RoundPolicy(depth)) if adaptive else None
        begin=time.perf_counter();r=core.start(ids,max_new_tokens=n,cache_prompt=cache)
        pre=time.perf_counter() if r.offset>=ids.shape[1]-1 else None
        out=[]
        while not r.done:
            e=r.step();out+=e['token_ids']
            if e['phase']=='prefill' and r.offset>=ids.shape[1]-1:pre=time.perf_counter()
        end=time.perf_counter();pre=pre or begin
        rec=dict(label=label,prompt_tokens=ids.shape[1],cached_tokens=r.cached_tokens,
                 chunk=chunk,depth=depth,adaptive=adaptive,prefill_seconds=pre-begin,
                 generation_seconds=end-pre,tps=n/(end-pre),tokens=out,profile=r.profile,
                 policy_cost=r.policy.cost if r.policy else None)
        result['runs'].append(rec);save()
        print('OPT_RUN',json.dumps({k:v for k,v in rec.items() if k not in ('tokens','profile')}),flush=True)
        return rec
    if os.environ.get('DS41_OPT_SKIP_CACHE','0')=='1':
        prior=state/'optimization-cache-qualified.json'
        cached=json.loads(prior.read_text())
        assert cached['cache_pass'] and [m['model_id'] for m in cached['manifests']]==[m['model_id'] for m in result['manifests']]
        result.update(cache_pass=True,cache_receipt_reused=str(prior),prefix_stats=cached['prefix_stats'])
        save()
    else:
        messages=[dict(role='user',content=('Blue birds nest near the river. The winter flood changes the shoreline.\n'*280)+
                             '\nSummarize the observations and their limitations in plain language.')]
        base=prompt(messages)
        cold=run(base,'cache_cold')
        warm=run(base,'cache_store',cache=True)
        hit=run(base,'cache_repeat',cache=True)
        assert cold['tokens']==warm['tokens']==hit['tokens'],'repeated cached tokens differ'
        assert hit['cached_tokens']>=base.shape[1]-8,'long repeat prefix missed'
        text=tok.decode(torch.tensor(warm['tokens']),decode_special_tokens=True)
        follow=prompt(messages+[dict(role='assistant',content=text),dict(role='user',content='Which observation would you investigate next, and why?')])
        reference=run(follow,'follow_cold')
        reused=run(follow,'follow_cached',cache=True)
        assert reused['tokens']==reference['tokens'],'follow-up cached tokens differ'
        assert reused['cached_tokens']>=base.shape[1]-16,'assistant-boundary prefix missed'
        thinking=prompt(messages,thinking=True)
        reference=run(thinking,'thinking_cold',n=16)
        reused=run(thinking,'thinking_cached',n=16,cache=True)
        assert reused['tokens']==reference['tokens'],'thinking-toggle cache differs'
        assert reused['cached_tokens']==0,'changed system reasoning budget reused stale prefix'
        again=run(thinking,'thinking_repeat',n=16,cache=True)
        assert again['tokens']==reference['tokens'] and again['cached_tokens']>1000
        result['cache_pass']=True;result['prefix_stats']=pool.stats();save()
    core.clear_prefix_cache()
    # Same independent input, no retained snapshots: change only prefill rows.
    long=prompt([dict(role='user',content=('A blue bird landed beside the river. Record the observation carefully.\n'*550)+
                    '\nExplain the limits of these repeated observations in detail.')])
    chunks=[]
    reference=None
    for chunk in (256,512,1024):
        rec=run(long,'prefill_'+str(chunk),n=16,chunk=chunk)
        if reference is None:reference=rec['tokens']
        rec['output_matches_256']=rec['tokens']==reference
        chunks.append(rec);save()
    best=min((r for r in chunks if r['output_matches_256']),key=lambda r:r['prefill_seconds'])
    result['selected_chunk']=best['chunk'];save()
    fixed=[];adaptive=[]
    for i,text in enumerate(('Explain why tides vary through the day in plain language. Give a useful example.',
                            'Write a Python function that merges overlapping intervals. Explain its edge cases briefly.')):
        ids=prompt([dict(role='user',content=text)])
        serial=run(ids,'serial'+str(i),n=96,chunk=best['chunk'],depth=0)
        old=json.loads((state/'qualification.json').read_text())
        prior=next(r['tokens'] for r in old['runs'] if r['label']==f'prompt{i}' and r['depth']==0)
        assert serial['tokens'][:len(prior)]==prior,'new kernel changed qualified serial outputs'
        f=run(ids,'fixed'+str(i),n=96,chunk=best['chunk'])
        a=run(ids,'adaptive'+str(i),n=96,chunk=best['chunk'],adaptive=True)
        assert f['tokens']==a['tokens']==serial['tokens'],'adaptive/fixed/serial target mismatch'
        fixed.append(f);adaptive.append(a)
    ft=192/sum(r['generation_seconds'] for r in fixed)
    at=192/sum(r['generation_seconds'] for r in adaptive)
    result.update(fixed_tps=ft,adaptive_tps=at,adaptive_selected=at>ft*1.01,
                  all_tokens_exact=True,completed=time.time())
    core.clear_prefix_cache();pipe.depth=3;save()
    print('OPTIMIZATION_PASS',json.dumps({k:v for k,v in result.items() if k not in ('runs','manifests')}),flush=True)
    return result


def qualify_prefill(pipe,tok,state):
    """Batched prefill may change rounding: check facts and exact same-chunk reuse."""
    import re
    from tools.ds41 import ds41_chat
    state=Path(state)
    prior=json.loads((state/'optimization-frontier-qualified.json').read_text())
    assert prior['all_tokens_exact'] and prior['cache_pass']
    assert [c.manifest['model_id'] for c in pipe.clients]==[m['model_id'] for m in prior['manifests']]
    result=dict(start=time.time(),runs=[],manifests=[c.manifest for c in pipe.clients],
                frontier_receipt=str(state/'optimization-frontier-qualified.json'))
    def save(): (state/'optimization-final-prefill.json').write_text(json.dumps(result,indent=2))
    mark=lambda text:int(tok.encode(text,add_bos=False,encode_special_tokens=True).flatten()[0])
    core=GreedyEngine(pipe,vocab_size=tok.actual_vocab_size,prefill_chunk=256,
          adaptive_drafting=False,prefix_pool=PrefixPool(pipe,user_id=mark('<｜User｜>'),assistant_id=mark('<｜Assistant｜>')))
    pipe.depth=3
    content=('The reference card says PASSKEY = 472913. There are 17 red items and 32 blue items.\n'+
             ('A blue bird landed beside the river. This is unrelated background information.\n'*500)+
             '\nReturn only JSON with keys "passkey" (a string) and "total" (the total red plus blue items). Use the reference card.')
    messages=[dict(role='user',content=content)]
    def run(messages,chunk,cache,label,total):
        ids=tok.encode(ds41_chat.render_chat(messages,thinking=False),add_bos=False,encode_special_tokens=True).cpu()
        core.prefill_chunk=chunk;t=time.perf_counter()
        r=core.start(ids,max_new_tokens=48,cache_prompt=cache,stop_ids=[mark('<｜end▁of▁sentence｜>')])
        pre=time.perf_counter() if r.offset>=ids.shape[1]-1 else None;out=[]
        while not r.done:
            e=r.step();out+=e['token_ids']
            if e['phase']=='prefill' and r.offset>=ids.shape[1]-1:pre=time.perf_counter()
        text=tok.decode(torch.tensor(out),decode_special_tokens=True)
        m=re.search(r'\{[^{}]*\}',text);answer=json.loads(m.group()) if m else None
        assert answer==dict(passkey='472913',total=total),(label,text)
        rec=dict(label=label,chunk=chunk,prompt_tokens=ids.shape[1],cached_tokens=r.cached_tokens,
                 prefill_seconds=(pre or t)-t,total_seconds=time.perf_counter()-t,tokens=out,text=text,answer=answer)
        result['runs'].append(rec);save();print('PREFILL_QUALITY',json.dumps(rec),flush=True)
        return rec
    baseline=run(messages,256,False,'reference256',49)
    candidate=run(messages,1024,True,'candidate1024',49)
    warm=run(messages,1024,True,'repeat1024',49)
    assert candidate['tokens']==warm['tokens'] and warm['cached_tokens']>6000
    following=messages+[dict(role='assistant',content=candidate['text']),dict(role='user',content='Add 5 more items to the total. Return the same JSON keys and passkey, with the updated total.')]
    reference=run(following,1024,False,'follow_cold1024',54)
    reused=run(following,1024,True,'follow_cached1024',54)
    assert reused['tokens']==reference['tokens'] and reused['cached_tokens']>6000
    speedup=baseline['prefill_seconds']/candidate['prefill_seconds']
    result.update(quality_smoke_pass=True,cache_tokens_exact=True,prefill_speedup=speedup,
                  selected_chunk=1024 if speedup>1.05 else 256,adaptive_selected=False,
                  fixed_tps=prior['fixed_tps'],adaptive_tps=prior['adaptive_tps'],completed=time.time(),
                  caveat='Different prefill shapes can change wording; same quant and factual smoke, not a broad quality benchmark')
    core.clear_prefix_cache();save();return result
