"""Cooperative greedy request lifecycle over committed speculative transactions.

One lock covers each entire distributed transaction. Cancellation is observed
at its boundary, never by interrupting a prepare/commit or recycling live pages.
This is an opt-in engine adapter, not an HTTP server or prefix-sharing cache.
"""
from threading import Event, Lock
from uuid import uuid4
from time import perf_counter, monotonic
import os

from .speculative import greedy_prefix


class Request:
    def __init__(self, engine, name, ids, max_new_tokens, stop_ids):
        self.engine, self.name, self.ids = engine, name, ids
        self.max_new_tokens, self.stop_ids = max_new_tokens, stop_ids
        self.cancelled = Event()
        self.offset = self.generated = 0
        self.current = ids[:, -1:]
        self.done = False
        self.finish_reason = None
        self.cache_prompt = True
        self.cached_tokens = 0
        self.profile = dict(accepted_draft_tokens=0,rejected_draft_tokens=0,draft_steps=0,
                            verify_positions=0,rounds=0,draft_seconds=0.,verify_seconds=0.,commit_seconds=0.)
        self.policy = engine.policy_factory() if engine.policy_factory else None
        self.sampler = None
        self.worker_sampling = False
        self.speculative = True
        self.guard = None
        self.loop_detected = False

    def decide(self, proposals, output):
        remaining = self.max_new_tokens - self.generated
        if self.worker_sampling:
            decision = output['decision']
            emitted = decision['emitted']
            accepted = decision['accepted']
            draft = proposals[0].tolist()
            if (type(emitted) is not list or not 1 <= len(emitted) <= min(remaining, len(draft) + 1)
                    or any(type(t) is not int or not 0 <= t < self.engine.vocab_size for t in emitted)
                    or type(accepted) is not int or not 0 <= accepted <= min(len(draft), len(emitted))
                    or emitted[:accepted] != draft[:accepted]
                    or len(emitted) - accepted > 1
                    or (accepted < min(len(draft), len(emitted)) and emitted[accepted] == draft[accepted])
                    or any(t in self.stop_ids for t in emitted[:-1])
                    or (accepted == len(emitted) and len(emitted) < remaining and emitted[-1] not in self.stop_ids)):
                raise RuntimeError('invalid worker sampling decision')
            return decision
        if self.sampler is None:
            return greedy_prefix(proposals[0].tolist(), output['tokens'][0].tolist(), remaining)
        from .sampling import sampled_prefix
        return sampled_prefix(proposals[0].tolist(), output['logits'][0, :, :self.engine.vocab_size],
                              remaining, self.sampler, self.stop_ids)

    def guard_tokens(self, emitted):
        if self.guard:
            for i, token in enumerate(emitted):
                if token in self.stop_ids:
                    break
                if self.guard.push(token):
                    self.loop_detected = True
                    return emitted[:i+1]
        return emitted

    def cancel(self):
        """Thread-safe signal; the next committed boundary releases this slot."""
        self.cancelled.set()

    def step(self):
        return self.engine.step(self)

    def close(self):
        if not self.done:
            self.cancel()
            self.step()

    def __iter__(self):
        try:
            while not self.done:
                yield self.step()
        finally:
            self.close()


class GreedyEngine:
    def __init__(self, pipeline, *, vocab_size, prefill_chunk=256, adaptive_drafting=True, prefix_pool=None, policy_factory=None):
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError('positive vocabulary size required')
        limits = [c.manifest.get('max_rows') for c in pipeline.clients]
        if any(type(n) is not int or n < pipeline.depth + 1 for n in limits):
            raise ValueError('workers must support the complete verification block')
        if type(prefill_chunk) is not int or not 1 <= prefill_chunk <= min(limits):
            raise ValueError('prefill chunk exceeds worker geometry')
        self.pipeline, self.vocab_size = pipeline, vocab_size
        self.prefill_chunk = prefill_chunk
        self.adaptive_drafting = adaptive_drafting
        self.lock, self.active, self.closed = Lock(), {}, False
        self.prefix_pool,self.policy_factory = prefix_pool,policy_factory
        setting = os.environ.get('DS41_CLUSTER_IDLE_RELEASE', '0')
        if setting not in ('0', '1'):
            raise ValueError('DS41_CLUSTER_IDLE_RELEASE must be 0 or 1')
        self.idle_release = setting == '1'
        if self.idle_release and any(c.manifest.get('mode') != 'cached-ds41-spec-v1' for c in pipeline.clients):
            raise ValueError('idle release requires DS41 stages')
        self.last_idle_release = None
        self.idle_release_report = None

    def start(self, ids, *, max_new_tokens, stop_ids=(), cache_prompt=True, temperature=0.0, top_p=1.0, seed=None, speculative=True, repetition_guard=False):
        import torch
        from .sampling import make_sampler, RepetitionGuard
        sampler = make_sampler(temperature, top_p, seed)
        if type(speculative) is not bool or type(repetition_guard) is not bool:
            raise ValueError('speculative and repetition_guard must be booleans')
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or
                ids.device.type != 'cpu' or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or
                not bool(((ids >= 0) & (ids < self.vocab_size)).all())):
            raise ValueError('one nonempty CPU int64 sequence within the vocabulary is required')
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('positive integer output budget required')
        if ids.shape[1] + max_new_tokens > self.pipeline.sessions.context:
            raise ValueError('prompt plus output reservation exceeds session context')
        if not isinstance(stop_ids, (list, tuple, set, frozenset)) or any(
                type(t) is not int or not 0 <= t < self.vocab_size for t in stop_ids):
            raise ValueError('invalid stop token IDs')
        with self.lock:
            if self.closed or self.pipeline.failed:
                raise RuntimeError('engine is closed; fresh workers required')
            # Reject capacity locally: sending an invalid open poisons workers.
            if len(self.pipeline.sessions.items) >= self.pipeline.sessions.slots:
                raise ValueError('no free request slot')
            name = 'req_' + uuid4().hex
            request = Request(self, name, ids.clone(), max_new_tokens, frozenset(stop_ids))
            try:
                self.pipeline.open(name)
                request.cache_prompt = cache_prompt
                request.sampler = sampler
                if sampler is not None and os.environ.get('DS41_CLUSTER_WORKER_SAMPLING') == '1':
                    if self.pipeline.clients[-1].manifest.get('mode') != 'cached-ds41-spec-v1':
                        raise ValueError('worker sampling requires a DeepSeek final stage')
                    ack = self.pipeline.clients[-1].call('configure_sampling', dict(
                        session=name, temperature=temperature, top_p=top_p,
                        seed=sampler.generator.initial_seed(), budget=max_new_tokens,
                        vocab=self.vocab_size, stop_ids=sorted(stop_ids)))
                    if ack != {'configured': True}:
                        raise RuntimeError('worker sampling configuration failed')
                    request.worker_sampling = True
                request.speculative = speculative
                request.guard = RepetitionGuard() if repetition_guard else None
                if self.prefix_pool: self.prefix_pool.begin(request)
            except BaseException:
                self._fail()
                raise
            self.active[name] = request
            return request

    def _event(self, request, tokens=(), phase='generation'):
        return dict(token_ids=list(tokens), phase=phase, done=request.done,
                    finish_reason=request.finish_reason, prompt_tokens=request.ids.shape[1],
                    completion_tokens=request.generated)

    def clear_prefix_cache(self):
        with self.lock:
            if self.prefix_pool: self.prefix_pool.clear()

    def _finish(self, request, reason):
        self.pipeline.release(request.name)
        request.done, request.finish_reason = True, reason
        self.active.pop(request.name)
        now = monotonic()
        if (self.idle_release and not self.active and not self.pipeline.sessions.items and
                (self.last_idle_release is None or now - self.last_idle_release >= 60)):
            # The caller holds the engine lock; no open, prepare or commit can
            # race this paired idle check. Stage telemetry failures are data.
            self.last_idle_release = now
            self.idle_release_report = [client.call('idle_release', {}) for client in self.pipeline.clients]

    def _fail(self):
        self.closed = True
        self.pipeline.close()
        for request in self.active.values():
            request.done, request.finish_reason = True, 'error'
        self.active.clear()

    def step(self, request):
        import torch
        with self.lock:
            if request.engine is not self:
                raise ValueError('request belongs to another engine')
            if request.done:
                return self._event(request)
            if self.closed or self.pipeline.failed:
                self._fail()
                raise RuntimeError('engine is closed; fresh workers required')
            if self.active.get(request.name) is not request:
                raise ValueError('request is not active')
            try:
                if request.cancelled.is_set():
                    self._finish(request, 'cancelled')
                    return self._event(request)
                prefix = request.ids.shape[1] - 1
                if request.offset < prefix:
                    end = min(prefix, request.offset + self.prefill_chunk)
                    if self.prefix_pool: end = self.prefix_pool.end(request,end)
                    ids = request.ids[:, request.offset:end]
                    self.pipeline.prepare(request.name, ids)
                    self.pipeline.commit(request.name, ids.shape[1])
                    if self.prefix_pool: self.prefix_pool.remember(request,ids,prefill=True)
                    request.offset = end
                    if request.cancelled.is_set():
                        self._finish(request, 'cancelled')
                    return self._event(request, phase='prefill')
                state = self.pipeline.sessions.items[request.name]
                want = request.policy.choose() if request.policy else self.pipeline.depth
                depth = min(want, self.pipeline.sessions.context - state.length - 1)
                # A single-token prompt has no target carry yet. Near the cache
                # boundary, R1 also avoids requesting rows beyond owned pages.
                if not request.speculative or state.length < getattr(self.pipeline, 'minimum_draft_position', 1):
                    depth = 0
                began = perf_counter()
                proposals = (self.pipeline.draft(request.name, request.current, depth,
                             adaptive=self.adaptive_drafting)['tokens']
                             if depth else torch.empty((1, 0), dtype=torch.int64))
                drafted = perf_counter()
                ids = torch.cat((request.current, proposals), dim=1)
                output = self.pipeline.prepare(request.name, ids, speculative=bool(depth), return_logits=request.sampler is not None)
                verified = perf_counter()
                decision = request.decide(proposals, output)
                emitted = request.guard_tokens(decision['emitted'])
                stop = next((i for i, token in enumerate(emitted) if token in request.stop_ids), None)
                if stop is not None:
                    emitted = emitted[:stop + 1]
                self.pipeline.commit(request.name, len(emitted))
                committed = perf_counter()
                if self.prefix_pool: self.prefix_pool.remember(request,ids[:,:len(emitted)])
                p=request.profile;p['rounds']+=1;p['verify_positions']+=ids.shape[1]
                p['draft_seconds']+=drafted-began;p['verify_seconds']+=verified-drafted;p['commit_seconds']+=committed-verified
                if proposals.numel():
                    accepted=min(decision['accepted'],len(emitted))
                    p['draft_steps']+=1;p['accepted_draft_tokens']+=accepted
                    p['rejected_draft_tokens']+=proposals.numel()-accepted
                if request.policy: request.policy.observe(depth,committed-began,len(emitted))
                if request.cancelled.is_set():
                    self._finish(request, 'cancelled')
                    return self._event(request)
                request.current = torch.tensor([[emitted[-1]]], dtype=torch.int64)
                request.generated += len(emitted)
                if stop is not None:
                    self._finish(request, 'stop')
                    visible = emitted[:-1]
                else:
                    visible = emitted
                    if request.loop_detected:
                        self._finish(request, 'repetition')
                    elif request.generated == request.max_new_tokens:
                        self._finish(request, 'length')
                return self._event(request, visible)
            except BaseException:
                self._fail()
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                for request in list(self.active.values()):
                    self._finish(request, 'cancelled')
            finally:
                self._fail()
