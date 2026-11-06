"""Opt-in greedy Qwen MTP: target prepare, accepted-prefix commit, local draft KV.

No sampling, prefix sharing, cancellation recovery or reconnect continuation.
An uncertain write or partial distributed commit requires fresh worker state.
"""
from .stage import CachedQwenStage, CachedGlmStage
from .worker import Pipeline
from .session import Sessions

MODE='cached-qwen-spec-v1'
GLM_MODE='cached-glm-spec-v1'
DS41_MODE='cached-ds41-spec-v1'

def greedy_prefix(drafts,predictions,remaining):
    """Choose the committed input prefix and emitted outputs of one verify block.

    The first input is the current unprocessed token. The final emitted token
    stays unprocessed, including when the request ends inside an accepted block.
    """
    if type(remaining) is not int or remaining<1:
        raise ValueError('positive integer output budget required')
    if type(drafts) not in (list,tuple) or type(predictions) not in (list,tuple):
        raise ValueError('explicit token sequences required')
    if len(drafts)>7 or len(predictions)!=len(drafts)+1 or any(type(x) is not int or x<0 for x in (*drafts,*predictions)):
        raise ValueError('invalid greedy verification geometry/tokens')
    accepted=0
    while accepted<len(drafts) and drafts[accepted]==predictions[accepted]:accepted+=1
    emitted=(list(drafts[:accepted])+[predictions[accepted]])[:remaining]
    return {'accepted':accepted,'committed':len(emitted),'emitted':emitted}

class TargetBinding:
    """Stable weak-reference owner for local embedding/head shared by MTP."""
    loaded_tp=False

class SpeculativeQwenStage(CachedQwenStage):
    protocol_mode=MODE
    uses_qwen_tier=True
    def __init__(self,model,begin,end,*,speculation_depth=7,**kwargs):
        if type(speculation_depth) is not int or not 1<=speculation_depth<=7:
            raise ValueError('supported speculative depth is 1..7')
        self.depth=speculation_depth;self.final=end==len(model.modules)
        self.embed=model.modules[0] if self.final else None
        self.calls={};self.draft_pending={};self.carry={}
        self.draft_model=self.draft_cache=self.binding=None
        super().__init__(model,begin,end,history=speculation_depth,**kwargs)
        self.manifest.update(mode=self.protocol_mode,max_history=speculation_depth,greedy_only=True)

    def load(self):
        super().load()
        if not self.final:return
        import torch
        from exllamav3 import Model,Cache
        from exllamav3.cache import CacheLayer_quant
        # Only duplicate the CPU embedding; the target head/mixer stay shared.
        if self.manifest['begin']!=0:self.embed.load(torch.device('cpu'))
        binding=TargetBinding();binding.config=self.model.config
        binding.modules=([self.embed]+self.model.modules) if self.manifest['begin'] else self.model.modules
        binding.logit_layer_idx=len(binding.modules)-1
        self.binding=binding
        self.draft_model=Model.from_config(self.model.config,component='mtp')
        kw=dict(layer_type=CacheLayer_quant,k_bits=8,v_bits=8) if self.manifest['cache_bits']==8 else {}
        self.draft_cache=Cache(self.draft_model,max_num_tokens=self.sessions.context*self.sessions.slots,
                               max_batch_size=self.sessions.slots,max_history=self.depth,**kw)
        self.draft_model.load(device=self.device,progressbar=False)
        self.draft_model.attach_to(binding)
        from exllamav3 import Tokenizer
        self.vocab=Tokenizer.from_config(self.model.config).actual_vocab_size
        self.draft_model.vocab_limit=self.vocab

    def cache_params(self,name,session,past,ids):
        params=super().cache_params(name,session,past,ids)
        speculative,_=self.calls[name]
        params.update(recurrent_history=speculative,dflash_verify=speculative)
        if self.final:
            if self.uses_qwen_tier:
                params['export_state_norm_keys']={self.model.modules[-2].key}
            else:
                params.update(self.draft_model.draft_verifier_params)
            params['export_states']=[]
        return params

    def draft_params(self,name,past,hidden):
        import torch
        slot=self.sessions.items[name].slot;pages=self.sessions.context//256
        return dict(attn_mode='flash_attn',cache=self.draft_cache,target_hidden=hidden,
                    cache_seqlens=torch.tensor([past],dtype=torch.int32),
                    block_table=torch.arange(slot*pages,(slot+1)*pages,dtype=torch.int32).view(1,pages))

    def draft(self,args):
        import torch
        if not self.final or set(args)!={'session','step','past','input_ids','depth','adaptive'}:
            raise ValueError('draft requires the final stage and explicit round geometry')
        name=args['session'];session=self.sessions.items[name];ids=args['input_ids'];depth=args['depth']
        if (session.pending is not None or name in self.draft_pending or
            type(args['step']) is not int or type(args['past']) is not int or
            (args['step'],args['past'])!=(session.step,session.length)):
            raise ValueError('stale/pending draft round')
        if type(depth) is not int or not 1<=depth<=self.depth or type(args['adaptive']) is not bool:
            raise ValueError('invalid draft budget')
        if not isinstance(ids,torch.Tensor) or ids.dtype!=torch.int64 or tuple(ids.shape)!=(1,1):
            raise ValueError('draft requires one current token')
        if self.carry.get(name) is None or session.length+depth+1>self.sessions.context:
            raise ValueError('draft requires committed target carry and context capacity')
        rule=None
        if self.uses_qwen_tier:
            from tools.qwen.spec_policy import TierRule
            rule=TierRule(maxd=depth);rule.begin()
        elif args['adaptive']:
            raise ValueError('GLM currently requires fixed-depth drafting')
        token=ids;hidden=self.carry[name];proposals=[];probabilities=[]
        for index in range(depth):
            params=self.draft_params(name,session.length+index,hidden)
            hidden=self.draft_model.forward(token,params)
            token=self.draft_model.sample_from_state(hidden,params).view(1,1)
            # The shared embedding is CPU-resident. Native Generator stages
            # every sampled ID back to CPU before the next MTP forward too.
            token=torch.where(token<self.vocab,token,torch.zeros_like(token)).cpu()
            p=None
            if self.uses_qwen_tier:
                if 'draft_p' not in params:raise RuntimeError('qualified draft2s probability is unavailable')
                p=float(params['draft_p'].item())
                if not 0<=p<=1:raise RuntimeError('invalid draft2s probability')
            keep,more=rule.step(index,p) if args['adaptive'] else (True,True)
            if keep:proposals.append(token.cpu().clone());probabilities.append(p)
            if not keep or not more:break
        if not proposals:raise RuntimeError('draft policy produced empty block')
        self.draft_pending[name]=(session.step,session.length,ids.cpu().clone())
        return {'tokens':torch.cat(proposals,dim=1),'probabilities':probabilities}

    def commit_prefix(self,args):
        import torch
        if set(args)!={'session','step','accepted'}:raise ValueError('invalid speculative commit')
        name=args['session'];accepted=self.sessions.validate_commit(name,args['step'],args['accepted'])
        session=self.sessions.items[name];ids,params=self.pending_recurrent[name]
        speculative,_=self.calls[name];rows=ids.shape[1];past=session.length
        if accepted!=rows and not speculative:raise ValueError('partial commit requires recorded history')
        draft=self.draft_pending.get(name)
        if draft is not None and (draft[:2]!=(session.step,past) or not torch.equal(ids[:,:1].cpu(),draft[2])):
            raise ValueError('target verification differs from pending draft input')
        from exllamav3.cache.recurrent_util import advance_recurrent_states
        advance_recurrent_states(ids,params,self.model)
        if speculative:self.states[name].rewind(rows-accepted)  # rewind(0) normalizes full acceptance too
        if self.states[name].position!=past+accepted:raise RuntimeError('target recurrent prefix differs')
        if self.final:
            hidden=params['export_states'][-1]
            if hidden.shape[:2]!=ids.shape:raise RuntimeError('incomplete target hidden export')
            if draft is None:
                carry=self.carry.get(name)
                if carry is None:carry=torch.zeros_like(hidden[:,:1])
                shifted=torch.cat((carry,hidden[:,:accepted-1]),dim=1)
                draft_params=self.draft_params(name,past,shifted)
                # Match native Job.prefill; accepted-token catch-up below must
                # retain its normal prefill path, as in Generator.iterate_gen.
                import os
                if os.environ.get('EXL3_PF_SKIP','0')=='1':draft_params['kv_only']=True
                self.draft_model.prefill(ids[:,:accepted],draft_params)
            elif accepted>1:
                self.draft_model.prefill(ids[:,1:accepted],self.draft_params(name,past+1,hidden[:,:accepted-1]))
            self.carry[name]=hidden[:,accepted-1:accepted].clone()
        result=self.sessions.commit(name,args['step'],accepted)
        self.pending_recurrent.pop(name);self.calls.pop(name);self.draft_pending.pop(name,None)
        return result

    def _execute_recurrent(self,op,args):
        try:
            if self.poisoned:raise RuntimeError('speculative worker is poisoned; restart required')
            if type(args) is not dict:raise ValueError('invalid speculative operation')
            if op=='draft':return self.draft(args)
            if op=='commit':return self.commit_prefix(args)
            if op=='prepare':
                required={'session','step','past','input_ids','activation','last_tokens_only','speculative','return_logits'}
                if set(args)!=required or type(args['speculative']) is not bool or type(args['return_logits']) is not bool:
                    raise ValueError('invalid speculative prepare')
                rows=args['input_ids'].shape[1]
                if args['speculative'] and not 1<=rows<=self.depth+1:raise ValueError('verify exceeds recurrent history')
                name=args['session'];self.calls[name]=(args['speculative'],args['return_logits'])
                clean={key:value for key,value in args.items() if key not in ('speculative','return_logits')}
                output=super()._execute_recurrent(op,clean)
                if not self.final:return output
                import torch
                if not bool(torch.isfinite(output).all()):raise RuntimeError('nonfinite target logits')
                result={'tokens':output[:,:,:self.vocab].argmax(-1).cpu()}
                if args['return_logits']:result['logits']=output.cpu()
                return result
            if op=='release' and args.get('session') in self.draft_pending:
                raise ValueError('cannot release pending draft state')
            result=super()._execute_recurrent(op,args)
            if op=='open':self.carry[args['session']]=None
            if op=='release':self.carry.pop(args['session'],None)
            return result
        except BaseException:
            self.poisoned=True
            raise

class SpeculativeGlmStage(SpeculativeQwenStage, CachedGlmStage):
    """GLM native MTP with its own hidden tap and explicit fixed draft depth.

    Reuses transaction/cache ownership; no Qwen acceptance policy or draft2s
    probability is assumed. GPU parity must qualify this opt-in adapter.
    """
    protocol_mode=GLM_MODE
    uses_qwen_tier=False


class SpeculativePipeline(Pipeline):
    def __init__(self,clients,*,mode=MODE):
        if mode not in (MODE,GLM_MODE,DS41_MODE):raise ValueError('unsupported speculative architecture mode')
        super().__init__(clients,mode=mode)
        self.minimum_draft_position = 2 if mode == DS41_MODE else 1
        geometry={(c.manifest.get('context'),c.manifest.get('sessions'),c.manifest.get('cache_bits'),c.manifest.get('max_history')) for c in clients}
        if len(geometry)!=1:raise ValueError('speculative cache geometry differs')
        context,slots,_,depth=geometry.pop()
        if type(depth) is not int or not 1<=depth<=7:raise ValueError('invalid speculative depth')
        self.sessions=Sessions(context,slots);self.depth=depth;self.failed=False

    def require_live(self):
        if self.failed:raise RuntimeError('pipeline is closed; restart required')

    def open(self,name):
        self.require_live()
        try:
            expected=self.sessions.open(name)
            for client in self.clients:
                if client.call('open',{'session':name})!=expected:raise RuntimeError('open acknowledgement differs')
        except BaseException:self.close();raise

    def draft(self,name,current,depth,adaptive=True):
        self.require_live()
        try:
            state=self.sessions.items[name]
            if state.pending is not None:raise ValueError('draft while target prepare pending')
            return self.clients[-1].call('draft',{'session':name,'step':state.step,'past':state.length,
                                                'input_ids':current,'depth':depth,'adaptive':adaptive})
        except BaseException:self.close();raise

    def prepare(self,name,ids,*,speculative=False,return_logits=False):
        self.require_live()
        try:
            state=self.sessions.items[name];step,past=state.step,state.length
            self.sessions.begin(name,step,past,ids.shape[1]);x=None
            for client in self.clients:
                x=client.call('prepare',dict(session=name,step=step,past=past,input_ids=ids,activation=x,
                    last_tokens_only=ids.shape[1] if speculative else 1,speculative=speculative,return_logits=return_logits))
            self.sessions.mark_ready(name)
            return x
        except BaseException:self.close();raise

    def commit(self,name,accepted):
        self.require_live()
        try:
            state=self.sessions.items[name];step=state.step
            accepted=self.sessions.validate_commit(name,step,accepted)
            expected={'length':state.length+accepted,'step':step+1}
            for client in self.clients:
                if client.call('commit',{'session':name,'step':step,'accepted':accepted})!=expected:
                    raise RuntimeError('prefix commit acknowledgement differs')
            self.sessions.commit(name,step,accepted)
            return expected
        except BaseException:self.close();raise

    def release(self,name):
        self.require_live()
        try:
            if self.sessions.items[name].pending is not None:raise ValueError('release during verification')
            for client in self.clients:
                if client.call('release',{'session':name})!={'released':True}:raise RuntimeError('release acknowledgement differs')
            self.sessions.release(name)
        except BaseException:self.close();raise

    def close(self):
        self.failed=True;super().close()
