"""Choose measured seconds per emitted token, with occasional bounded probes."""


class RoundPolicy:
    def __init__(self, maximum=3):
        self.depths = tuple(dict.fromkeys((min(3,maximum),0,min(2,maximum))))
        self.cost, self.count, self.rounds = {}, {}, 0

    def choose(self):
        # Two samples of each path avoid choosing from one cold kernel launch.
        for d in self.depths:
            if self.count.get(d,0)<2: return d
        best = min(self.cost,key=self.cost.get)
        if self.rounds % 16 == 0:
            return self.depths[(self.rounds//16) % len(self.depths)]
        return best

    def observe(self, depth, seconds, emitted):
        if emitted <= 0: return
        value = seconds/emitted
        old = self.cost.get(depth,value)
        self.cost[depth] = .65*old+.35*value
        self.count[depth] = self.count.get(depth,0)+1
        self.rounds += 1
