"""Chunk-stage pipeline with coordinated prefix checkpoints."""
from tools.multinode.generation import GreedyEngine
from .prefill_pipeline import prefill_pipeline
from .prefill_pipeline import verify_jobs, draft_verify_jobs
from tools.multinode.speculative import greedy_prefix
import asyncio
import os
import time
import torch


class PrefillEngine(GreedyEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        value = os.environ.get('DS41_CLUSTER_PREFILL_WINDOW', '4')
        try:
            self.prefill_window = int(value)
        except ValueError as exc:
            raise ValueError('invalid prefill window') from exc
        if not 1 <= self.prefill_window <= 16:
            raise ValueError('prefill window must be between 1 and 16')
        def enabled(name):
            value = os.environ.get(name, '0')
            if value not in ('0', '1'):
                raise ValueError(f'{name} must be 0 or 1')
            return value == '1'
        self.paired_draft_overlap = enabled('DS41_CLUSTER_PAIRED_DRAFT_OVERLAP')
        native = enabled('DS41_CLUSTER_PREFILL_NATIVE')
        self.prefill_state_only = native or enabled('DS41_CLUSTER_PREFILL_STATE_ONLY')
        self.prefill_ahead = native or enabled('DS41_CLUSTER_PREFILL_AHEAD')
        self.prefill_tail = native or enabled('DS41_CLUSTER_PREFILL_HID_ROWS_128')

    async def async_step(self, request):
        if not hasattr(self,'waiting'):
            self.waiting=[];self.flush_scheduled=False
        future=asyncio.get_running_loop().create_future()
        self.waiting.append((request,future))
        if not self.flush_scheduled:
            self.flush_scheduled=True
            asyncio.get_running_loop().call_later(.001,lambda:asyncio.create_task(self.flush()))
        return await future

    async def flush(self):
        batch=self.waiting[:2];del self.waiting[:2];self.flush_scheduled=False
        try:
            events=await asyncio.to_thread(self.step_many,[r for r,_ in batch])
            for (_,f),e in zip(batch,events):
                if not f.done():f.set_result(e)
        except BaseException as error:
            for _,f in batch:
                if not f.done():f.set_exception(error)
        if self.waiting and not self.flush_scheduled:
            self.flush_scheduled=True
            asyncio.get_running_loop().call_soon(lambda:asyncio.create_task(self.flush()))

    def step_many(self, requests):
        eligible=(len(requests)==2 and requests[0].name != requests[1].name
            and type(self.pipeline.depth) is int and 1 <= self.pipeline.depth <= 7) and all(r.speculative and not r.done and not r.cancelled.is_set() and r.policy is None
            and r.offset>=r.ids.shape[1]-1 and r.name in self.pipeline.sessions.items
            and 2<=self.pipeline.sessions.items[r.name].length<self.pipeline.sessions.context-max(8,self.pipeline.depth+1) for r in requests)
        if not eligible:return [self.step(r) for r in requests]
        with self.lock:
            try:
                jobs=[];proposals={};began=time.perf_counter()
                for r in requests:
                    if self.closed or self.active.get(r.name) is not r:raise RuntimeError('request inactive')
                if self.paired_draft_overlap:
                    jobs,proposals,predictions=draft_verify_jobs(self.pipeline,requests)
                    drafted=began;verified=time.perf_counter()
                else:
                    for r in requests:
                        proposal=self.pipeline.draft(r.name,r.current,self.pipeline.depth,adaptive=False)['tokens']
                        proposals[r.name]=proposal
                        jobs.append((r.name,torch.cat((r.current,proposal),dim=1)))
                    drafted=time.perf_counter()
                    predictions=verify_jobs(self.pipeline,jobs, return_logits=any(r.sampler is not None for r in requests));verified=time.perf_counter()
                events=[]
                for r,(name,ids) in zip(requests,jobs):
                    proposal=proposals[name]
                    decision=r.decide(proposal,predictions[name])
                    emitted=r.guard_tokens(decision['emitted']);stop=next((i for i,t in enumerate(emitted) if t in r.stop_ids),None)
                    if stop is not None:emitted=emitted[:stop+1]
                    self.pipeline.commit(name,len(emitted))
                    if self.prefix_pool:self.prefix_pool.remember(r,ids[:,:len(emitted)])
                    p=r.profile;p['rounds']+=1;p['verify_positions']+=ids.shape[1];p['draft_seconds']+=drafted-began;p['verify_seconds']+=verified-drafted
                    p['draft_steps']+=1;p['accepted_draft_tokens']+=min(decision['accepted'],len(emitted));p['rejected_draft_tokens']+=proposal.numel()-min(decision['accepted'],len(emitted))
                    if r.cancelled.is_set():self._finish(r,'cancelled');events.append(self._event(r));continue
                    r.current=torch.tensor([[emitted[-1]]]);r.generated+=len(emitted)
                    visible=emitted
                    if stop is not None:self._finish(r,'stop');visible=emitted[:-1]
                    elif r.loop_detected:self._finish(r,'repetition')
                    elif r.generated==r.max_new_tokens:self._finish(r,'length')
                    events.append(self._event(r,visible))
                return events
            except BaseException:self._fail();raise

    def step(self, request):
        if request.done or request.offset >= request.ids.shape[1]-1:
            return super().step(request)
        with self.lock:
            if request.cancelled.is_set():
                self._finish(request,'cancelled')
                return self._event(request)
            if self.closed or self.pipeline.failed:
                raise RuntimeError('engine closed')
            start=request.offset;offset=start;chunks=[]
            try:
                window = min(self.prefill_window, 4) if len(self.active) > 1 else self.prefill_window
                for _ in range(window):
                    end=min(request.ids.shape[1]-1,offset+self.prefill_chunk)
                    request.offset=offset
                    if self.prefix_pool:end=self.prefix_pool.end(request,end)
                    chunks.append(request.ids[:,offset:end]);offset=end
                    checkpoint=self.prefix_pool and (offset in request.prefill_marks or offset%self.prefix_pool.grid==0)
                    if checkpoint or offset>=request.ids.shape[1]-1:break
            finally:request.offset=start
            try:
                _, completed = prefill_pipeline(self.pipeline,request.name,chunks,
                                                cancelled=request.cancelled.is_set,
                                                state_only=self.prefill_state_only,
                                                prefetch_ahead=self.prefill_ahead,
                                                hid_rows=128 if self.prefill_tail else None,
                                                return_completed=True)
                chunks = chunks[:completed]
                offset = start + sum(chunk.shape[1] for chunk in chunks)
                for chunk in chunks:
                    if self.prefix_pool:self.prefix_pool.remember(request,chunk,prefill=True)
                request.offset=offset
                if request.cancelled.is_set():self._finish(request,'cancelled')
                return self._event(request,phase='prefill')
            except BaseException:
                self._fail();raise
