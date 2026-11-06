"""Guarded fast path for the CPU float32, top-p-only Gumbel sampler."""
import torch
from tools.ds41.dspark import truncated_probs


def sample(logits, temperature, top_p, generator):
    if logits.device.type != "cpu" or not (temperature > 0 and 0 < top_p < 1):
        raise ValueError("candidate requires CPU temperature sampling with top-p")
    probs = torch.softmax(logits.reshape(-1) / max(temperature, 1e-5), dim=-1, dtype=torch.float32)
    noise = torch.empty_like(probs).exponential_(1, generator=generator)
    scores = probs / noise
    top, ids = scores.topk(min(2, scores.numel()))
    # Normalizing before division can perturb nearly tied scores. Delegate these
    # cases to the reference path using exactly the already drawn noise.
    if len(top) == 2 and top[0].isfinite() and top[1] > torch.finfo(probs.dtype).tiny and top[0] > top[1] * 1.00001:
        winner = int(ids[0])
        p = probs[winner]
        # Count all equal probabilities conservatively. When this bound is
        # below the cutoff, stable sorting must retain this winner.
        through = float(torch.sum(probs * (probs >= p), dtype=torch.float64))
        eps = torch.finfo(probs.dtype).eps
        # Worst-case roundoff of a positive FP32 prefix sum + subtraction.
        margin = 2 * (len(probs) + 2) * eps
        if through + margin < top_p and p >= torch.finfo(probs.dtype).tiny:
            return winner, True
    retained = truncated_probs(logits, temperature, top_p)
    return int((retained / noise).argmax()), False
