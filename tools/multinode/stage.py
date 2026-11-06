"""First EXL3 integration gate: uncached dense text module spans only.

Weights outside the assigned range are never loaded. Cached/recurrent models,
MTP, vision and relayering require additional lifecycle protocols, not pickle.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SUPPORTED = {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"}


def source_fingerprint(root):
    digest = hashlib.sha256()
    paths = list((Path(root) / "exllamav3").rglob("*")) + list((Path(root) / "tools/multinode").rglob("*.py"))
    for p in sorted(paths):
        if p.is_file() and p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".hip"}:
            digest.update(p.relative_to(root).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(p.read_bytes()).digest())
    return digest.hexdigest()


class DenseStage:
    architectures = SUPPORTED
    allow_recurrent = False

    def __init__(self, model, begin, end, *, device, model_id, source_id, max_rows=2048):
        if model.config.architecture not in self.architectures:
            raise NotImplementedError("pilot supports uncached dense Llama/Qwen2/Qwen3 only; recurrent/MTP/vision paths are not implemented")
        if model.loaded_tp or model.config.layer_map or (model.recurrent_state_cls is not None and not self.allow_recurrent):
            raise NotImplementedError("TP, relayering and recurrent state are unsupported; on ROCm the uncached pilot requires EXL3_KVWIN=0 before constructing models")
        if any(ref() is not None for ref in model.cache_weakrefs.values()):
            raise ValueError("construct stages before attaching any cache")
        if not 0 <= begin < end <= len(model.modules) or max_rows < 1:
            raise ValueError("invalid module span or row budget")
        self.manifest = dict(protocol=1, mode="uncached-dense-v1", model_id=model_id,
                             source_id=source_id, begin=begin, end=end,
                             total_modules=len(model.modules),
                             module_keys=[m.key for m in model.modules[begin:end]],
                             max_rows=max_rows)
        self.model, self.device, self.max_rows = model, device, max_rows
        # Creating the architecture builds metadata; selecting modules happens
        # BEFORE load(), so unowned weight tensors are never materialized.
        model.modules = model.modules[begin:end]
        model.fwd_modules = [(m, 0, i) for i, m in enumerate(model.modules, start=begin)]
        model.modules_dict = None
        # The recurrent guard can populate cached full-model layer lists. Drop
        # them after pruning so cache allocation/prefetch owns only this span.
        for key in ('_get_cache_layers','_get_recurrent_layers','_get_checkpoint_layers','_get_prefetch_layers'):
            model.__dict__.pop(key,None)

    def valid_activation(self,x,ids):
        import torch
        return isinstance(x,torch.Tensor) and x.ndim==3 and tuple(x.shape[:2])==tuple(ids.shape) and x.shape[-1]==self.model.config.hidden_size and x.dtype in (torch.float16,torch.bfloat16,torch.float32)

    def load(self):
        self.model.load(device=self.device, progressbar=False)

    def forward(self, request):
        return self._forward(request, {})

    def finalize_activation(self,x,params):
        return x

    def _forward(self, request, params):
        import torch
        if set(request) != {"input_ids", "activation", "last_tokens_only"}:
            raise ValueError("unsupported parameters; cache/session/MTP handles cannot cross this protocol")
        ids, x, last = request["input_ids"], request["activation"], request["last_tokens_only"]
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 2:
            raise ValueError("input_ids must be a batch x sequence int64 tensor")
        if not 1 <= ids.shape[0] <= 8 or not 1 <= ids.shape[1] <= self.max_rows:
            raise ValueError("input exceeds stage batch/sequence budget")
        if type(last) is not int or not 1 <= last <= ids.shape[1]:
            raise ValueError("invalid output row limit")
        if self.manifest["begin"] == 0:
            if x is not None:
                raise ValueError("first stage accepts input IDs only")
        elif not self.valid_activation(x,ids):
            raise ValueError("invalid dense activation shape/dtype")
        with torch.inference_mode():
            params = dict(params, last_tokens_only=last, input_ids=ids)
            prepared = self.model.prepare_inputs(ids, params)
            for module in getattr(self.model,'_get_prefetch_layers',[]):
                module.prefetch(prepared,params)
            if x is None:
                x = prepared
            for module, instance, _ in self.model.fwd_modules:
                params["layer_instance"] = instance
                if module.caps.get("logits_output"):
                    x = x[..., -last:, :].contiguous()
                x = module.prepare_for_device(x, params)
                x = module.forward(x, params)
                if x is None:
                    raise RuntimeError("unexpected prefill-only stop in uncached forward")
            return self.finalize_activation(x,params)


class CachedDenseStage(DenseStage):
    """Experimental fixed-page dense sessions; explicit prepare/commit per step.

    Any execution error poisons this worker. Reconnect/retry requires fresh
    worker state; committed positions are never guessed after a partial write.
    """
    def __init__(self, *args, context=512, sessions=2, cache_bits=16, history=0, **kwargs):
        from .session import Sessions
        self.sessions=Sessions(context,sessions)
        self.poisoned=False
        super().__init__(*args,**kwargs)
        from exllamav3 import Cache
        # Query cache layers only after module selection. Never allocate KV
        # storage for the other node's modules.
        self.model.__dict__.pop('_get_cache_layers',None)
        if cache_bits not in (8,16):raise ValueError('unsupported cache width')
        cache_args={}
        if cache_bits==8:
            from exllamav3.cache import CacheLayer_quant
            cache_args=dict(layer_type=CacheLayer_quant,k_bits=8,v_bits=8)
        self.cache=Cache(self.model,max_num_tokens=context*sessions,max_batch_size=sessions,max_history=history,**cache_args)
        self.manifest.update(mode='cached-dense-v1',context=context,sessions=sessions,cache_bits=cache_bits)

    def cache_params(self,name,session,past,ids):
        import torch
        pages=self.sessions.context//256
        return {'attn_mode':'flash_attn','cache':self.cache,
                'block_table':torch.arange(session.slot*pages,(session.slot+1)*pages,dtype=torch.int32).view(1,pages),
                'cache_seqlens':torch.tensor([past],dtype=torch.int32)}

    def execute(self, op, args):
        import torch
        if self.poisoned:raise RuntimeError('cached worker is poisoned; restart required')
        if type(args) is not dict:raise ValueError('invalid cached operation')
        if op=='open' and set(args)=={'session'}:
            return self.sessions.open(args['session'])
        if op=='commit' and set(args)=={'session','step'}:
            return self.sessions.commit(args['session'],args['step'])
        if op=='release' and set(args)=={'session'}:
            return self.sessions.release(args['session'])
        if op!='prepare' or set(args)!={'session','step','past','input_ids','activation','last_tokens_only'}:
            raise ValueError('unsupported cached operation')
        ids=args['input_ids']
        if not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1:
            raise ValueError('cached pilot supports one sequence per session step')
        session=self.sessions.begin(args['session'],args['step'],args['past'],ids.shape[1])
        try:
            params=self.cache_params(args['session'],session,args['past'],ids)
            output=self._forward({key:args[key] for key in ('input_ids','activation','last_tokens_only')},params)
            self.sessions.mark_ready(args['session'])
            return output
        except BaseException:
            self.poisoned=True
            raise

    def forward(self, request):
        raise ValueError('cached stages require explicit prepare/commit')


class CachedQwenStage(CachedDenseStage):
    """Experimental Qwen4Exp text span: exact FP32 streams and node-local GDN/PLE.

    No speculative history/rejection, vision aliases or arbitrary relayering.
    Recurrent positions advance only on commit; uncertain writes poison worker.
    """
    architectures={'Qwen4ExpForConditionalGeneration'}
    allow_recurrent=True

    def __init__(self, model, begin, end, **kwargs):
        self.stream_input = 2<=begin<=len(model.modules)-2
        self.states={}
        self.pending_recurrent={}
        self.boundary_flushes=0
        super().__init__(model,begin,end,**kwargs)
        # Keep the same transaction protocol; architecture identity is included
        # explicitly, so a Qwen span cannot be paired with a dense span.
        self.manifest.update(architecture=model.config.architecture,hc_mult=model.config.hc_mult)

    def valid_activation(self,x,ids):
        import torch
        if not self.stream_input:return super().valid_activation(x,ids)
        return (isinstance(x,torch.Tensor) and x.ndim==4 and tuple(x.shape[:2])==tuple(ids.shape)
                and x.shape[2:]==(self.model.config.hc_mult,self.model.config.hidden_size) and x.dtype==torch.float32)

    def cache_params(self,name,session,past,ids):
        params=super().cache_params(name,session,past,ids)
        state=self.states[name]
        if state.position!=past:raise ValueError('local recurrent position differs from committed prefix')
        params['recurrent_states']=[state]
        self.pending_recurrent[name]=(ids,params)
        return params

    @staticmethod
    def flush_pending(x,pending):
        from exllamav3.modules.hyperconnections import HyperConnection
        return HyperConnection.flush_pending(x,pending)

    def finalize_activation(self,x,params):
        # Prefill fuses a residual update into the NEXT block's normalization.
        # Its tensor tuple is local scratch, not part of the wire activation.
        # Materialize the update before crossing a stage boundary.
        pending=params.pop('hc_pending',None)
        if pending is not None:
            self.flush_pending(x,pending)
            self.boundary_flushes=getattr(self,'boundary_flushes',0)+1
        return x

    def execute(self,op,args):
        import torch
        # Recurrent buffers are allocated as inference tensors during load.
        with torch.inference_mode():
            return self._execute_recurrent(op,args)

    def _execute_recurrent(self,op,args):
        try:
            result=super().execute(op,args)
            name=args['session']
            if op=='open':
                self.states[name]=self.cache.get_new_state()
            elif op=='commit':
                from exllamav3.cache.recurrent_util import advance_recurrent_states
                ids,params=self.pending_recurrent.pop(name)
                advance_recurrent_states(ids,params,self.model)
                if self.states[name].position!=result['length']:
                    raise RuntimeError('recurrent position failed to commit')
            elif op=='release':
                self.states.pop(name).free()
            return result
        except BaseException:
            self.poisoned=True
            raise


class CachedGlmStage(CachedQwenStage):
    """GLM5Next mHC/KDA/MLA text stages with independent local DSA indexers.

    Shared-index selection across a partition is deliberately unsupported.
    Only complete transformer-block boundaries are accepted for this first gate.
    """
    architectures = {'Glm5NextForConditionalGeneration'}

    def __init__(self, model, begin, end, **kwargs):
        config = model.config
        if len(config.indexer_types) < config.num_hidden_layers or any(
                kind != 'full' for kind in config.indexer_types[:config.num_hidden_layers]):
            raise NotImplementedError('GLM cluster requires independent full indexers on every layer')
        trunk_end = model.first_block_idx + config.num_hidden_layers
        if begin != 0 and not model.first_block_idx <= begin < trunk_end:
            raise ValueError('GLM stage must begin at input or a complete transformer block')
        if end != len(model.modules) and not model.first_block_idx < end <= trunk_end:
            raise ValueError('GLM stage must end after a complete block or output head')
        super().__init__(model, begin, end, **kwargs)
        self.stream_input = begin != 0


def load_stage(model_dir, begin, end, device, weights_manifest, max_rows=2048, *, cache_context=0, cache_sessions=2, cache_bits=16, experimental_qwen=False, experimental_glm=False, speculation_depth=0):
    from .identity import verify
    model_id = verify(model_dir, weights_manifest)
    from exllamav3 import Config, Model
    config = Config.from_directory(model_dir)
    allowed=SUPPORTED | CachedQwenStage.architectures if experimental_qwen else SUPPORTED
    if experimental_glm:allowed=allowed | CachedGlmStage.architectures
    if config.architecture not in allowed:
        raise NotImplementedError(f"unsupported pilot architecture {config.architecture}")
    if config.architecture in CachedQwenStage.architectures | CachedGlmStage.architectures and not cache_context:
        raise ValueError('hybrid model requires explicit cached session ownership')
    model = Model.from_config(config)
    root = Path(__file__).resolve().parents[2]
    stage_cls = (CachedGlmStage if config.architecture in CachedGlmStage.architectures else
                 CachedQwenStage if config.architecture in CachedQwenStage.architectures else
                 CachedDenseStage if cache_context else DenseStage)
    cache_args = dict(context=cache_context,sessions=cache_sessions,cache_bits=cache_bits) if cache_context else {}
    if speculation_depth:
        from .speculative import SpeculativeQwenStage, SpeculativeGlmStage
        if experimental_qwen and config.architecture in CachedQwenStage.architectures:
            stage_cls=SpeculativeQwenStage
        elif experimental_glm and config.architecture in CachedGlmStage.architectures:
            stage_cls=SpeculativeGlmStage
        else:
            raise ValueError('speculation requires an explicit cached Qwen or GLM adapter')
        cache_args['speculation_depth']=speculation_depth
    stage = stage_cls(model, begin, end, device=device,
                       model_id=model_id,
                       source_id=source_fingerprint(root), max_rows=max_rows, **cache_args)
    stage.load()
    return stage
