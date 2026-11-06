"""Opt-in async adapter for Kyojin's existing Qwen HTTP/template/tool handlers.

Only greedy MTP is implemented. Requests never share prefixes; every finished
or cancelled request releases its native recurrent state and fixed KV slot.
"""
import asyncio
import time


def stream_prefix(text, stops, final=False):
    """Hold incomplete stop strings and incomplete trailing UTF-8 decoding."""
    hits = [text.find(stop) for stop in stops if stop and stop in text]
    if hits:
        return text[:min(hits)], True
    if final:
        return text, False
    text = text.rstrip('\ufffd')
    held = max((n for stop in stops for n in range(1, min(len(stop), len(text) + 1))
                if text.endswith(stop[:n])), default=0)
    return text[:-held] if held else text, False


class QwenClusterEngine:
    supports_sampling = False
    supports_vision = False
    spec_on = True
    slot_store = prefix_disk = None

    def __init__(self, core, tokenizer, eos, *, bad_request=ValueError):
        import torch
        self.core, self.tokenizer, self.eos = core, tokenizer, tuple(eos)
        self.torch, self.bad_request = torch, bad_request
        self.ctx = core.pipeline.sessions.context
        self.vocab_size = core.vocab_size
        self.sessions = core.pipeline.sessions.slots
        self.ndt = core.pipeline.depth
        self.draft_model = object()  # HTTP draft-window reservation; no remote model handle
        self.last_stats = {}

    def prompt_ids(self, prompt):
        return self.tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)

    def encode_ids(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_bos=add_special_tokens,
                                     encode_special_tokens=True).flatten().tolist()

    def count_tokens(self, text):
        return int(self.prompt_ids(text).numel())

    def token_str(self, token_id):
        return self.tokenizer.decode(self.torch.tensor([token_id]), decode_special_tokens=True)

    def spec_stats(self):
        return dict(speculative=True, clustered=True, greedy_only=not self.supports_sampling,
                    prefix_cache=self.core.prefix_pool is not None, healthy=not self.core.closed and not self.core.pipeline.failed)

    async def reset_prefix_cache(self):
        # No prefix state survives a request, so a cache_salt change needs no
        # network mutation. The existing HTTP admission gate drains requests.
        if self.core.prefix_pool:
            await asyncio.to_thread(self.core.clear_prefix_cache)
        return None

    async def generate(self, prompt, stats=None, *, max_tokens, sampling, stop,
                       images=None, speculative=True, cache_prompt=True, cancel=None,
                       seed=None, reasoning_first=False, thinking_budget=None,
                       loop_guard=None, loop_final=None, logprobs=None,
                       prompt_ids=None, ignore_eos=False, repetition_guard=True):
        if not self.supports_sampling and sampling.get('temperature', 0) != 0 and sampling.get('top_k') != 1:
            raise self.bad_request('experimental cluster supports greedy decoding only (temperature=0)')
        if images or (not speculative and not self.supports_sampling) or logprobs is not None or thinking_budget is not None or loop_guard or loop_final:
            raise self.bad_request('experimental cluster does not support vision, plain decoding, logprobs or loop/thinking guards')
        if cancel is not None and cancel.is_set():
            return
        ids = (self.torch.tensor([prompt_ids], dtype=self.torch.int64) if prompt_ids is not None
               else self.prompt_ids(prompt))
        stats = {} if stats is None else stats
        stats.update(prompt_tokens=int(ids.numel()), cached_tokens=0, token_ids=[], new_tokens=0)
        request = pending = start_task = None
        tokens, sent, first = [], '', None
        begin = time.perf_counter()
        stopped = False
        try:
            start_task = asyncio.create_task(asyncio.to_thread(self.core.start, ids,
                max_new_tokens=max_tokens, stop_ids=() if ignore_eos else self.eos,
                cache_prompt=cache_prompt, **(dict(temperature=sampling.get('temperature', 0),
                top_p=sampling.get('top_p', 1), seed=seed, speculative=speculative,
                repetition_guard=repetition_guard) if self.supports_sampling else {})))
            request = await asyncio.shield(start_task)
            while not request.done:
                if cancel is not None and cancel.is_set():
                    request.cancel()
                step = getattr(self.core, 'async_step', None)
                pending = asyncio.create_task(step(request) if step else asyncio.to_thread(request.step))
                event = await asyncio.shield(pending)
                pending = None
                if cancel is not None and cancel.is_set():
                    request.cancel()
                    break
                if event['phase'] == 'generation' and first is None:
                    first = time.perf_counter()
                tokens.extend(event['token_ids'])
                text = self.tokenizer.decode(self.torch.tensor(tokens, dtype=self.torch.int64),
                                             decode_special_tokens=True) if tokens else ''
                visible, stopped = stream_prefix(text, stop, final=request.done)
                if not visible.startswith(sent):
                    raise RuntimeError('token decoder revised already emitted text')
                delta = visible[len(sent):]
                sent = visible
                if delta:
                    yield delta
                if stopped:
                    request.cancel()
                    break
            if request.finish_reason == 'repetition':
                raise RuntimeError('Generation stopped: extreme repetition detected; completed tools must not be replayed')
            if request.finish_reason == 'error':
                raise RuntimeError('distributed request failed; fresh workers required')
        finally:
            # Shield the in-flight transaction: releasing its pages before its
            # thread finishes would corrupt a different request reusing them.
            async def cleanup():
                nonlocal request
                if request is None and start_task is not None:
                    request = await start_task
                if request is not None:
                    request.cancel()
                    if pending is not None:
                        await pending
                    await asyncio.to_thread(request.close)
            await asyncio.shield(cleanup())
            end = time.perf_counter()
            stats.update(token_ids=tokens, new_tokens=request.generated if request else 0,
                         time_prefill=(first or end) - begin,
                         time_generate=end - (first or end),
                         eos_reason='max_new_tokens' if request and request.finish_reason == 'length' and not stopped
                         else 'stop_string' if stopped else 'stop_token')
            if request is not None:
                stats.update(request.profile,cached_tokens=request.cached_tokens)
            self.last_stats = stats

    async def close(self):
        await asyncio.to_thread(self.core.close)
