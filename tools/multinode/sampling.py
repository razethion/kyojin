"""Exact target sampling with deterministic draft proposals and isolated RNG."""
import math
import os
import secrets
from collections import deque


def validate_sampling(temperature, top_p, seed):
    if (type(temperature) not in (float, int) or not math.isfinite(temperature)
            or not 0 <= temperature <= 2):
        raise ValueError('temperature must be finite and in [0, 2]')
    if (type(top_p) not in (float, int) or not math.isfinite(top_p)
            or not 0 < top_p <= 1):
        raise ValueError('top_p must be finite and in (0, 1]')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('seed must be an integer in [0, 2**63)')


def make_sampler(temperature, top_p, seed):
    validate_sampling(temperature, top_p, seed)
    if temperature == 0:
        return None
    import torch
    from tools.ds41.dspark import GumbelSampler
    generator = torch.Generator(device='cpu')
    generator.manual_seed(secrets.randbits(63) if seed is None else seed)
    sampler = GumbelSampler(temperature, generator, top_p=top_p)
    if top_p < 1 and os.environ.get('DS41_CLUSTER_FAST_TOP_P') == '1':
        from .fast_top_p import sample
        sampler.sample = lambda row: sample(row, temperature, top_p, generator)[0]
    return sampler


def sampled_prefix(drafts, logits, remaining, sampler, stop_ids=()):
    """Consume exactly one draw per emitted token, never the rejected tail.

    Drafts are deterministic, so sample p directly; matching draws accept the
    draft, and the first nonmatching draw is the correction. This is the same
    output law (and RNG sequence) as the ordinary target-only loop.
    """
    if remaining < 1 or logits.ndim != 2 or logits.shape[0] != len(drafts) + 1:
        raise ValueError('invalid sampled verification geometry')
    emitted = []
    accepted = 0
    for i in range(min(remaining, len(drafts) + 1)):
        token = sampler.sample(logits[i])
        emitted.append(token)
        matches = i < len(drafts) and token == drafts[i]
        accepted += int(matches)
        if token in stop_ids or not matches:
            break
    return dict(emitted=emitted, accepted=accepted, committed=len(emitted))


class RepetitionGuard:
    """Conservative exact periodic suffix guard, including hidden reasoning."""
    def __init__(self):
        self.tail = deque(maxlen=256)

    def push(self, token):
        self.tail.append(token)
        if len(self.tail) < 256:
            return False
        seq = list(self.tail)
        return any(all(seq[i] == seq[i-period] for i in range(period, 256))
                   for period in range(1, 17))
