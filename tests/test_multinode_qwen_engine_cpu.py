"""Async cancellation/text streaming over the transaction request engine."""
import asyncio
import threading
import unittest

import torch

from test_multinode_generation_cpu import engine
from tools.multinode.qwen_engine import QwenClusterEngine, stream_prefix


class Tokenizer:
    def encode(self, text, **kwargs):
        return torch.tensor([[4, 5]])
    def decode(self, ids, **kwargs):
        return ''.join(chr(65 + t) for t in ids.flatten().tolist())


class AsyncEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_string_split_across_rounds_does_not_leak(self):
        core, _ = engine()
        adapter = QwenClusterEngine(core, Tokenizer(), [])
        stats = {}
        pieces = [piece async for piece in adapter.generate('', stats=stats, max_tokens=20,
                    sampling={'temperature': 0}, stop=['NOP'])]
        self.assertEqual(''.join(pieces), 'GHIJKLM')
        self.assertEqual(stats['eos_reason'], 'stop_string')
        self.assertFalse(core.pipeline.sessions.items)
        await adapter.close()

    async def test_async_task_cancel_waits_for_commit_and_reuses_slot(self):
        core, stages = engine()
        adapter = QwenClusterEngine(core, Tokenizer(), [])
        entered, resume = threading.Event(), threading.Event()
        def hook(op):
            if op == 'prepare':
                entered.set()
                if not resume.wait(5): raise TimeoutError('test barrier expired')
        stages[-1].hook = hook
        async def consume():
            return [piece async for piece in adapter.generate('', max_tokens=4,
                        sampling={'temperature': 0}, stop=[])]
        task = asyncio.create_task(consume())
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            task.cancel()
            await asyncio.sleep(.01)
            self.assertFalse(task.done())
        finally:
            resume.set()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(core.pipeline.sessions.items)
        stages[-1].hook = None
        self.assertEqual(''.join(await consume()), 'GHIJ')
        await adapter.close()

    async def test_stream_close_releases_live_request(self):
        core, _ = engine()
        adapter = QwenClusterEngine(core, Tokenizer(), [])
        stream = adapter.generate('', max_tokens=20, sampling={'temperature': 0}, stop=[])
        self.assertEqual(await anext(stream), 'GHIJKLMN')
        await stream.aclose()
        self.assertFalse(core.pipeline.sessions.items)
        await adapter.close()

    async def test_unsupported_sampling_rejected_before_open(self):
        core, _ = engine()
        adapter = QwenClusterEngine(core, Tokenizer(), [])
        with self.assertRaises(ValueError):
            await anext(adapter.generate('', max_tokens=4, sampling={'temperature': .8}, stop=[]))
        self.assertFalse(core.pipeline.sessions.items)
        self.assertFalse(core.pipeline.failed)
        await adapter.close()

    async def test_two_streams_use_separate_slots(self):
        core, _ = engine()
        adapter = QwenClusterEngine(core, Tokenizer(), [])
        async def consume():
            return ''.join([piece async for piece in adapter.generate('', max_tokens=13,
                           sampling={'temperature': 0}, stop=[])])
        self.assertEqual(await asyncio.gather(consume(), consume()), ['GHIJKLMNOPQRS'] * 2)
        self.assertFalse(core.pipeline.sessions.items)
        await adapter.close()

    def test_incomplete_stop_and_unicode_suffix(self):
        self.assertEqual(stream_prefix('hello ST', ['STOP']), ('hello ', False))
        self.assertEqual(stream_prefix('hello STOPextra', ['STOP']), ('hello ', True))
        self.assertEqual(stream_prefix('hello\ufffd', []), ('hello', False))
        self.assertEqual(stream_prefix('hello\ufffd', [], True), ('hello\ufffd', False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
