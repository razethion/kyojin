"""Bounded, coordinated host snapshots; recurrent state never crosses the wire."""
from uuid import uuid4


class PrefixPool:
    def __init__(self, pipeline, *, budget=2 << 30, entries=16, grid=4096,
                 user_id=None, assistant_id=None):
        self.pipe, self.budget, self.limit, self.grid = pipeline, budget, entries, grid
        self.user_id, self.assistant_id = user_id, assistant_id
        self.items = []
        self.hits = self.reused = 0
        self.physical = [0 for _ in self.pipe.clients]

    def call(self, op, args):
        try:
            return [c.call(op, args) for c in self.pipe.clients]
        except BaseException:
            self.pipe.close()
            raise

    def begin(self, request):
        from exllamav3.dsv41.prefix_cache import marks_for
        ids = request.ids[0].tolist()
        request.fed = []
        # Split at every assistant boundary, so extending a conversation keeps
        # its earlier prefill segmentation. Reusing generated-token snapshots
        # would mix decode rounding with a cold batched prefill; keep prompt
        # checkpoints only until that separate numerical path is qualified.
        request.prefill_marks = sorted(set(marks_for(ids, 0, user_id=self.user_id,
                                          assistant_id=self.assistant_id) + [len(ids)-1] +
                                          [i+1 for i,t in enumerate(ids[:-1]) if t==self.assistant_id]))
        request.cached_tokens = 0
        if request.cache_prompt:
            matches = [e for e in self.items if e['n'] <= len(ids)-1 and ids[:e['n']] == e['keys']]
            if matches:
                e = max(matches, key=lambda e:e['n'])
                replies = self.call('prefix_restore', dict(session=request.name, key=e['key']))
                expected = dict(length=e['n'], step=0)
                if any(r != expected for r in replies):
                    self.pipe.close()
                    raise RuntimeError('prefix restore acknowledgement differs')
                self.pipe.sessions.items[request.name].length = e['n']
                request.offset = request.cached_tokens = e['n']
                request.fed = ids[:e['n']]
                self.items.remove(e); self.items.append(e)
                self.hits += 1; self.reused += e['n']

    def end(self, request, proposed_end):
        boundaries = [p for p in request.prefill_marks if request.offset < p <= proposed_end]
        grid = (request.offset // self.grid + 1) * self.grid
        if grid <= proposed_end: boundaries.append(grid)
        return min(boundaries, default=proposed_end)

    def remember(self, request, ids, prefill=False):
        request.fed.extend(ids[0].tolist())
        n = len(request.fed)
        if prefill and (n in request.prefill_marks or n % self.grid == 0): self.save(request)

    def save(self, request):
        if not request.cache_prompt or len(request.fed) < 2: return
        keys = request.fed
        if any(e['keys'] == keys for e in self.items): return
        key = uuid4().hex
        if self.limit < 1: return
        while True:
            estimates = self.call('prefix_estimate', dict(session=request.name, key=key))
            self.physical = [r['physical_bytes'] for r in estimates]
            over = any(r['physical_bytes'] + r['increment_bytes'] > self.budget for r in estimates)
            if len(self.items) < self.limit and not over: break
            if not self.items: return
            old = self.items.pop(0)
            self.call('prefix_drop', dict(key=old['key']))
            # Dropping the previous lineage checkpoint can remove the source
            # of reuse; estimate again rather than subtracting its logical size.
        rows = self.call('prefix_save', dict(session=request.name, key=key))
        if not all(r['saved'] for r in rows):
            self.call('prefix_drop', dict(key=key))
            self._measure()
            return
        e = dict(key=key, n=len(keys), keys=list(keys), sizes=[r['bytes'] for r in rows])
        self.items.append(e)
        self._measure()
        while self.items and (len(self.items)>self.limit or any(b>self.budget for b in self.physical)):
            old = self.items.pop(0)
            self.call('prefix_drop', dict(key=old['key']))
            self._measure()

    def _measure(self):
        self.physical = [r['physical_bytes'] for r in self.call('prefix_stats', {})]

    def clear(self):
        self.call('prefix_clear', {})
        self.items.clear()
        self.physical = [0 for _ in self.pipe.clients]

    def stats(self):
        return dict(entries=len(self.items), hits=self.hits, reused_tokens=self.reused,
                    bytes_per_node=list(self.physical),
                    logical_bytes_per_node=[sum(e['sizes'][i] for e in self.items)
                                            for i in range(len(self.pipe.clients))])
