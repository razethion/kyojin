"""Native DSV41 source-aligned stages and journaled DSpark transactions.

The mHC residual and delayed pre-mix travel together. Compressed attention
pools stay on their source's host, and all DSpark target taps stay on the last.
"""
from .session import Sessions

MODE = 'cached-ds41-spec-v1'


class DeepSeekStage:
    def __init__(self, model, *, model_id, source_id, context=262144,
                 sessions=2, depth=4, max_rows=256):
        cfg = model.cfg
        first, last = model.layer_span
        if not 1 <= depth <= 5 or max_rows < depth + 1:
            raise ValueError('invalid DSpark geometry')
        if first and first not in set(cfg.kv_source_layer_ids) & set(cfg.index_source_layer_ids):
            raise ValueError('shared attention source crosses partition')
        self.model, self.final, self.depth, self.max_rows = model, last == cfg.num_hidden_layers, depth, max_rows
        if self.final and (model.mtp is None or any(l < first for l in cfg.dspark_target_layer_ids)):
            raise ValueError('DSpark stages and target taps must all belong to final stage')
        self.sessions = Sessions(context, sessions)
        self.states, self.pending, self.carry, self.specs, self.draft_pending = {}, {}, {}, {}, {}
        self.poisoned = False
        self.trace_names, self.traces = set(), {}
        self.prefixes = {}
        self.prefix_lineage = {}
        self.worker_sampling = {}
        # Logical modules: embedding + expansion, trunk blocks, final norm + head.
        begin, end = (0 if first == 0 else first + 2), (last + 4 if self.final else last + 2)
        self.manifest = dict(protocol=1, mode=MODE, model_id=model_id, source_id=source_id,
            begin=begin, end=end, total_modules=cfg.num_hidden_layers + 4,
            architecture='DeepseekV41ForCausalLM', hc_mult=cfg.hc_mult,
            context=context, sessions=sessions, cache_bits=16, max_history=depth,
            max_rows=max_rows, greedy_only=True, layer_span=[first, last])

    def _discard_session(self, name):
        """Drop request-owned tensors without waiting for cyclic Python GC."""
        spec = self.specs.pop(name, None)
        if spec is not None:
            # PortSpec -> target/drafter -> PortSpec is a reference cycle. Its
            # state and hidden rows would otherwise keep GPU tensors alive
            # after release, beyond the subsequent idle empty_cache call.
            spec.st = spec.rows = spec.block = spec.spans = spec.keys = None
            spec.keys_run = []
            spec.store_now = None
            spec.target.s = spec.drafter.s = None
            spec.target = spec.drafter = spec.model = None
        state = self.states.pop(name, None)
        if state is not None:
            state.journal = None  # abandoned verification can never be resumed
        for owned in (self.carry, self.pending, self.draft_pending, self.traces):
            owned.pop(name, None)
        self.prefix_lineage.pop(name, None)
        self.worker_sampling.pop(name, None)
        self.trace_names.discard(name)

    def close(self):
        """Poison abandoned transactions and release this stage's session ownership.

        Static model weights/readers belong to the model, not this session store.
        Unlike normal release, final teardown also forgets host prefix snapshots.
        """
        self.poisoned = True
        names=set(self.sessions.items)|set(self.states)|set(self.specs)|set(self.carry)|set(self.pending)|set(self.draft_pending)
        for name in names:
            self._discard_session(name)
        self.sessions.items.clear()
        self.prefixes.clear();self.prefix_lineage.clear()
        self.traces.clear();self.trace_names.clear()

    def execute(self, op, args):
        import torch
        if self.poisoned:
            raise RuntimeError('DeepSeek stage is poisoned; restart required')
        try:
            with torch.inference_mode():
                return self._execute(op, args)
        except BaseException:
            self.poisoned = True
            raise

    def _execute(self, op, args):
        import torch
        if type(args) is not dict:
            raise ValueError('explicit operation arguments required')
        if op.startswith('prefix_'):
            return self.prefix_operation(op,args)
        if op == 'idle_release' and not args:
            return self.idle_release()
        if op == 'configure_sampling':
            from .sampling import make_sampler
            if not self.final or set(args) != {'session','temperature','top_p','seed','budget','vocab','stop_ids'}:
                raise ValueError('invalid worker sampler options')
            name = args['session']
            session = self.sessions.items[name]
            budget, vocab, stops = args['budget'], args['vocab'], args['stop_ids']
            if (session.step or session.pending is not None or name in self.worker_sampling
                    or type(budget) is not int or not 1 <= budget <= self.sessions.context
                    or type(vocab) is not int or not 1 <= vocab <= self.model.cfg.vocab_size
                    or type(stops) is not list or any(type(t) is not int or not 0 <= t < vocab for t in stops)):
                raise ValueError('invalid worker sampler request')
            sampler = make_sampler(args['temperature'], args['top_p'], args['seed'])
            if sampler is None:
                raise ValueError('worker sampler requires positive temperature')
            self.worker_sampling[name] = dict(sampler=sampler, remaining=budget, vocab=vocab, stops=stops, drawn=0)
            return {'configured': True}
        if op == 'debug':
            return self.debug(args)
        if op == 'capacity' and set(args) == {'session'}:
            name = args['session']; session = self.sessions.items[name]
            if session.length or session.pending is not None: raise ValueError('capacity probe requires an empty slot')
            self.states[name].ensure(self.sessions.context)
            from pathlib import Path
            fields = {l.split(':')[0]: int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
            return dict(cache_bytes=self.states[name].nbytes(), allocated_bytes=torch.cuda.memory_allocated()
                        if self.model.device.type == 'cuda' else 0, available_bytes=fields['MemAvailable'])
        if op == 'open' and set(args) == {'session'}:
            name = args['session']
            result = self.sessions.open(name)
            self.prefix_lineage.pop(name, None)
            self.states[name] = self.model.new_state(1, self.sessions.context)
            self.carry[name] = None
            if self.final:
                from exllamav3.dsv41.spec import PortSpec
                spec = self.specs[name] = PortSpec(self.model, self.sessions.context, self.depth)
                spec.st = self.states[name]
            return result
        if op == 'release' and set(args) == {'session'}:
            name = args['session']
            if name in self.draft_pending:
                raise ValueError('release with unverified draft')
            result = self.sessions.release(name)
            self._discard_session(name)
            return result
        if op == 'prepare':
            required = {'session', 'step', 'past', 'input_ids', 'activation',
                        'last_tokens_only', 'speculative', 'return_logits'}
            optional = {'state_only', 'ahead_ids', 'hid_rows'}
            if (not required <= set(args) or set(args) - required - optional or
                    type(args['speculative']) is not bool or type(args['return_logits']) is not bool):
                raise ValueError('invalid target prepare')
            name, ids, past = args['session'], args['input_ids'], args['past']
            if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 2 or
                    ids.shape[0] != 1 or not 1 <= ids.shape[1] <= self.max_rows or
                    bool(((ids < 0) | (ids >= self.model.cfg.vocab_size)).any())):
                raise ValueError('invalid input tokens')
            rows = ids.shape[1]
            state_only = args.get('state_only', False)
            ahead = args.get('ahead_ids')
            hid_rows = args.get('hid_rows')
            if type(state_only) is not bool:
                raise ValueError('invalid state-only selection')
            if args['speculative'] and (state_only or ahead is not None or hid_rows is not None):
                raise ValueError('prefill options are not valid for verification')
            if state_only and args['return_logits']:
                raise ValueError('state-only cannot return logits')
            if ahead is not None and (not isinstance(ahead, torch.Tensor) or ahead.dtype != torch.int64 or
                    ahead.ndim != 2 or ahead.shape[0] != 1 or not 1 <= ahead.shape[1] <= self.max_rows or
                    bool(((ahead < 0) | (ahead >= self.model.cfg.vocab_size)).any())):
                raise ValueError('invalid ahead tokens')
            if ahead is not None and past + rows + ahead.shape[1] > self.sessions.context:
                raise ValueError('ahead tokens exceed context')
            if hid_rows is not None and (type(hid_rows) is not int or
                    hid_rows < self.model.cfg.sliding_window or hid_rows > self.max_rows):
                raise ValueError('invalid hidden tail size')
            if args['speculative'] and rows > self.depth + 1:
                raise ValueError('verify exceeds draft budget')
            if args['last_tokens_only'] != (rows if args['speculative'] else 1):
                raise ValueError('invalid target logit selection')
            incoming = args['activation']
            if self.model.layer_span[0] == 0:
                if incoming is not None: raise ValueError('first stage accepts IDs only')
            else:
                if type(incoming) not in (list, tuple) or len(incoming) != 2:
                    raise ValueError('missing mHC carry')
                if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or not bool(t.isfinite().all()) for t in incoming):
                    raise ValueError('invalid mHC carry')
            state = self.sessions.begin(name, args['step'], past, rows)
            draft = self.draft_pending.get(name)
            if draft is not None and (draft[:2] != (state.step, past) or not torch.equal(draft[2], ids[:, :1].cpu())):
                raise ValueError('verify differs from pending anchor')
            st = self.states[name]
            if st.length != past: raise RuntimeError('cache position differs')
            if args['speculative']: st.begin_block(past, rows)
            taps = {} if name in self.trace_names else None
            options = {}
            if optional & set(args):
                options = dict(state_only=state_only and self.final,
                               ahead_ids=ahead if self.model.layer_span[0] == 0 else None,
                               hid_rows=hid_rows if self.final else None)
            output, hidden = self.model.forward_stage(ids, past, st, incoming,
                                                     all_logits=args['speculative'],taps=taps, **options)
            if taps is not None:
                self.traces.setdefault(name,[]).append(taps)
                self.traces[name] = self.traces[name][-8:]
            self.pending[name] = (rows, args['speculative'], hidden)
            self.sessions.mark_ready(name)
            if not self.final: return [t.cpu() for t in output]
            if state_only: return {'tokens': None}
            if not bool(output.isfinite().all()): raise RuntimeError('nonfinite target logits')
            if output.ndim == 2: output = output[:, None]
            sampling = self.worker_sampling.get(name)
            if sampling is not None and args['return_logits']:
                from .sampling import sampled_prefix
                decision = sampled_prefix(ids[0, 1:].tolist(), output[0, :, :sampling['vocab']].cpu(),
                                          sampling['remaining'], sampling['sampler'], sampling['stops'])
                sampling['drawn'] = len(decision['emitted'])
                return {'decision': decision}
            result = {'tokens': output.argmax(-1).cpu()}
            if args['return_logits']: result['logits'] = output.cpu()
            return result
        if op == 'commit' and set(args) == {'session', 'step', 'accepted'}:
            name = args['session']
            accepted = self.sessions.validate_commit(name, args['step'], args['accepted'])
            sampling = self.worker_sampling.get(name)
            if sampling is not None and sampling['drawn']:
                if accepted > sampling['drawn']:
                    raise ValueError('sampled commit exceeds emitted prefix')
                sampling['remaining'] -= accepted
                sampling['drawn'] = 0
            rows, speculative, hidden = self.pending[name]
            st = self.states[name]
            if accepted < rows:
                if not speculative: raise ValueError('partial prefill commit is unsupported')
                st.rollback(accepted)
            elif speculative:
                st.end_block()
            st.length = self.sessions.items[name].length + accepted
            if self.final:
                if hidden is None or (hidden.shape[1] < accepted and
                                      (speculative or accepted != rows or hidden.shape[1] < min(rows, self.model.cfg.sliding_window))):
                    raise RuntimeError('missing DSpark target taps')
                new = hidden[:, :accepted]
                old = self.carry[name]
                if name not in self.draft_pending and old is not None:
                    new = torch.cat((old, new), dim=1)
                self.carry[name] = new[:, -self.model.cfg.sliding_window:].clone()
            result = self.sessions.commit(name, args['step'], accepted)
            self.pending.pop(name)
            self.draft_pending.pop(name, None)
            return result
        if op == 'draft' and set(args) == {'session', 'step', 'past', 'input_ids', 'depth', 'adaptive'}:
            name = args['session']; session = self.sessions.items[name]
            ids, depth = args['input_ids'], args['depth']
            if (not self.final or session.pending is not None or name in self.draft_pending or
                    (args['step'], args['past']) != (session.step, session.length) or
                    type(depth) is not int or not 1 <= depth <= self.depth or args['adaptive'] is not False):
                raise ValueError('invalid DSpark round')
            if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or tuple(ids.shape) != (1, 1):
                raise ValueError('one anchor token required')
            if not 0 <= ids.item() < self.model.cfg.vocab_size or self.carry[name] is None:
                raise ValueError('invalid draft anchor or missing hidden carry')
            if session.length + depth + 1 > self.sessions.context:
                raise ValueError('draft exceeds context capacity')
            # DSpark evaluates a fixed block even when only one proposal is
            # requested. A chained pass needs another whole block of RoPE
            # positions. Finish the last few positions with target-only R1.
            block = self.model.cfg.dspark_block_size
            passes = -(-depth // (block - 1))
            if session.length + passes * block + 1 > self.sessions.context:
                return {'tokens': torch.empty((1,0),dtype=torch.int64), 'probabilities': []}
            out = self.specs[name].drafter.draft(int(ids.item()), self.carry[name][0], session.length - 1, want=depth, want_conf=False)
            self.draft_pending[name] = (session.step, session.length, ids.cpu().clone())
            return {'tokens': torch.tensor([out.tokens], dtype=torch.int64), 'probabilities': [None] * len(out.tokens)}
        raise ValueError('unsupported DeepSeek operation')

    def idle_release(self):
        """Release unused allocator blocks only when this worker is quiescent."""
        import torch
        if (self.sessions.items or self.pending or self.draft_pending or
                self.states or self.specs or self.carry):
            return {'released': False, 'reason': 'busy'}
        if self.model.device.type != 'cuda':
            return {'released': False, 'reason': 'not_cuda'}
        report = {'released': False}
        # Diagnostics must not poison a healthy worker when /proc or allocator
        # telemetry is unavailable. Live allocations and prefix tensors remain.
        def telemetry():
            out = {}
            try:
                out['allocated_bytes'] = int(torch.cuda.memory_allocated(self.model.device))
                out['reserved_bytes'] = int(torch.cuda.memory_reserved(self.model.device))
            except Exception:
                pass
            try:
                from pathlib import Path
                line = next(x for x in Path('/proc/meminfo').read_text().splitlines()
                            if x.startswith('MemAvailable:'))
                out['available_bytes'] = int(line.split()[1]) * 1024
            except Exception:
                pass
            return out
        report['before'] = telemetry()
        try:
            torch.cuda.synchronize(self.model.device)
            torch.cuda.empty_cache()
            report['released'] = True
        except Exception as exc:
            report['reason'] = type(exc).__name__
        report['after'] = telemetry()
        return report

    def prefix_operation(self,op,args):
        import re
        from exllamav3.dsv41 import prefix_cache as pc
        if op == 'prefix_stats' and not args:
            snaps = [item[0] for item in self.prefixes.values()]
            return {'physical_bytes': pc.unique_snapshot_bytes(snaps),
                    'logical_bytes': sum(s.nbytes for s in snaps), 'entries': len(snaps)}
        if op == 'prefix_clear' and not args:
            self.prefixes.clear();self.prefix_lineage.clear();return {'cleared':True}
        key=args.get('key')
        if not isinstance(key,str) or not re.fullmatch(r'[0-9a-f]{32}',key):raise ValueError('invalid prefix key')
        if op == 'prefix_drop' and set(args)=={'key'}:
            self.prefixes.pop(key,None)
            self.prefix_lineage = {name: prev for name,prev in self.prefix_lineage.items() if prev != key}
            return {'dropped':True}
        if set(args)!={'session','key'}:raise ValueError('invalid prefix operation')
        name=args['session'];session=self.sessions.items[name]
        if session.pending is not None or name in self.draft_pending:raise ValueError('prefix state is not committed')
        if op in ('prefix_estimate', 'prefix_save'):
            import os
            try: page_rows = int(os.environ.get('DS41_PFX_PAGE_ROWS', '0'))
            except ValueError as exc: raise ValueError('invalid prefix page size') from exc
            if not 0 <= page_rows <= 4096: raise ValueError('invalid prefix page size')
            previous = self.prefixes.get(self.prefix_lineage.get(name))
        if op == 'prefix_estimate':
            snaps = [item[0] for item in self.prefixes.values()]
            return {'upper_bound_bytes': pc.snapshot_bytes(self.model.cfg,self.states[name],session.length,self.carry[name]),
                    'increment_bytes': pc.snapshot_increment_bytes(self.model.cfg,self.states[name],session.length,
                        self.carry[name],previous=previous[0] if previous else None,page_rows=page_rows),
                    'physical_bytes': pc.unique_snapshot_bytes(snaps)}
        if op == 'prefix_save':
            if key in self.prefixes:raise ValueError('duplicate prefix key')
            if pc.mem_available_gib()<8 or len(self.prefixes)>=18:return {'saved':False,'bytes':0}
            options = {} if not page_rows else dict(page_rows=page_rows,
                previous=previous[0] if previous else None, trusted_previous=bool(previous))
            snap=pc.snapshot(self.model.cfg,self.states[name],session.length,(),self.carry[name],'cluster',**options)
            nxt=self.specs[name].drafter._next if self.final else 0
            self.prefixes[key]=(snap,nxt)
            self.prefix_lineage[name]=key
            return {'saved':True,'bytes':snap.nbytes}
        if op == 'prefix_restore':
            if session.length or session.step:raise ValueError('restore requires a fresh slot')
            snap,nxt=self.prefixes[key]
            self.prefix_lineage.pop(name, None)
            self.states[name]=pc.restore(self.model,snap,self.sessions.context)
            self.carry[name]=None if snap.hidden is None else snap.hidden.to(self.model.device)
            if self.final:
                self.specs[name].st=self.states[name];self.specs[name].drafter._next=nxt
            session.length=snap.n
            return {'length':snap.n,'step':0}
        raise ValueError('unknown prefix operation')

    def debug(self,args):
        """Authenticated, bounded diagnostics; never reached from HTTP requests."""
        import os
        import torch
        if args.get('action') == 'settings' and set(args)=={'action','engram'} and args['engram'] in ('0','1'):
            if self.sessions.items: raise ValueError('settings require no live sessions')
            os.environ['DSV41_VERIFY_ENGRAM']=args['engram']
            return {'engram':args['engram']}
        if args.get('action') == 'trace' and set(args)=={'action','session'}:
            name=args['session'];self.sessions.items[name]
            self.trace_names.add(name);self.traces[name]=[]
            return {'tracing':True}
        if args.get('action') != 'compare' or set(args)!={'action','left','right','rows','traces'}:
            raise ValueError('invalid diagnostic operation')
        left,right=args['left'],args['right'];rows=args['rows']
        a,b=self.states[left],self.states[right]
        if a.length!=b.length: raise ValueError('comparison lengths differ')
        if type(rows) is not int or not 1<=rows<=6 or type(args['traces']) is not bool:raise ValueError('invalid diagnostic window')
        diffs=[]
        def check(key,x,y):
            if x.shape!=y.shape:diffs.append(dict(key=key,shapes=[list(x.shape),list(y.shape)]));return
            if not torch.equal(x,y):
                d=torch.where(x==y,0,(x.float()-y.float()).abs());maximum=float(d.max())
                diffs.append(dict(key=key,count=int((x!=y).sum()),max_abs=maximum if maximum<float('inf') else 'nonfinite'))
        for l in range(*self.model.layer_span):check(f'L{l}.cache.win',a.win[l],b.win[l])
        for l in a.comp_kv:
            n=a.length//self.model.cfg.ratio(l)
            check(f'L{l}.cache.pool',a.comp_kv[l][:,:n],b.comp_kv[l][:,:n])
            check(f'L{l}.cache.index',a.idx_k[l][:,:n],b.idx_k[l][:,:n])
        for l in a.comp_kv_state:
            check(f'L{l}.cache.open_kv',a.comp_kv_state[l],b.comp_kv_state[l])
            check(f'L{l}.cache.open_score',a.comp_score_state[l],b.comp_score_state[l])
        if a.hasher is not None:check('hash_history',a.hasher.cache[:,:a.length],b.hasher.cache[:,:b.length])
        cache_count=len(diffs)
        if args['traces']:
            def combine(name,count):
                recent=self.traces.get(name,[])[-count:];out={}
                for trace in recent:
                    for key,t in trace.items():
                        if key.endswith(('.topk','.comp')): continue  # positional pool state is compared separately
                        if t.ndim>=3 and t.shape[0]==1:t=t.reshape(-1,*t.shape[2:])
                        out.setdefault(key,[]).append(t)
                return {k:torch.cat(v,dim=0)[:rows] for k,v in out.items()}
            x,y=combine(left,rows),combine(right,1)
            for key in sorted(x.keys()&y.keys(),key=lambda k:(int(k.split('.')[0][1:]),k)):
                check(key,x[key],y[key])
        return dict(length=a.length,cache_differences=cache_count,differences=diffs[:100])


def load_stage(model_dir, weights_manifest, first, last, *, device='cuda:0', context=262144,
               sessions=2, depth=4, max_rows=256):
    from pathlib import Path
    from .verified_files import verify
    from .stage import source_fingerprint
    from exllamav3.dsv41.pack import load
    model_id = verify(model_dir, weights_manifest, Path(weights_manifest).with_name('verified-stat.json'))
    model = load(model_dir, device=device, layer_span=(first, last), mtp=last == 40,
                 prefill_chunk=max_rows, log=lambda s: print(s, flush=True))
    print('MOE_BACKENDS', {l: {'K': [g.k_gu for g in be.store.groups], 'unified': be.uni is not None} for l, be in model.moe_backends.items()}, flush=True)
    return DeepSeekStage(model, model_id=model_id,
        source_id=source_fingerprint(Path(__file__).resolve().parents[2]), context=context,
        sessions=sessions, depth=depth, max_rows=max_rows)
