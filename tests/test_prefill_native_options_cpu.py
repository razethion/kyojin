"""Tiny native comparison for optional prefill paths; run in the torch environment."""
import unittest
import torch
from tests.test_multinode_ds41_cpu import DeepSeekTests
from tools.multinode.prefill_pipeline import prefill_pipeline


class NativePrefillOptionsTests(DeepSeekTests):
    def test_native_prefill_then_all_commit_lengths(self):
        # Independent live sessions start from the same weights and token history.
        stages, pipe = self.stages()
        chunks = [torch.tensor([[i % 500 for i in range(16)]]),
                  torch.tensor([[(i + 16) % 500 for i in range(16)]])]
        try:
            for name in ('baseline', 'native'):
                pipe.open(name)
            baseline = prefill_pipeline(pipe, 'baseline', chunks)
            native = prefill_pipeline(pipe, 'native', chunks, state_only=True,
                                      prefetch_ahead=True, hid_rows=self.cfg.sliding_window)
            self.assertTrue(torch.equal(baseline['tokens'], native['tokens']))
            for stage in stages:
                a, b = stage.states['baseline'], stage.states['native']
                self.assertEqual(a.length, b.length)
                for key in a.win:
                    self.assertTrue(torch.equal(a.win[key], b.win[key]))
                for key in a.comp_kv:
                    n = a.length // self.cfg.ratio(key)
                    self.assertTrue(torch.equal(a.comp_kv[key][:, :n], b.comp_kv[key][:, :n]))
                    self.assertTrue(torch.equal(a.idx_k[key][:, :n], b.idx_k[key][:, :n]))
                if stage.final:
                    self.assertTrue(torch.equal(stage.carry['baseline'], stage.carry['native']))
            for name in ('baseline', 'native'):
                pipe.release(name)
        finally:
            pipe.close()

    def test_prefix_restore_and_every_accepted_suffix(self):
        for accepted in range(1, 6):
            stages, pipe = self.stages()
            chunks = [torch.tensor([[i for i in range(16)]]),
                      torch.tensor([[i for i in range(16, 32)]])]
            try:
                pipe.open('source')
                prefill_pipeline(pipe, 'source', chunks, state_only=True,
                                 prefetch_ahead=True, hid_rows=self.cfg.sliding_window)
                key = 'a' * 32
                for client in pipe.clients:
                    estimate = client.call('prefix_estimate', dict(session='source', key=key))
                    self.assertGreater(estimate['upper_bound_bytes'], 0)
                    self.assertTrue(client.call('prefix_save', dict(session='source', key=key))['saved'])
                pipe.open('restored')
                for client in pipe.clients:
                    self.assertEqual(client.call('prefix_restore', dict(session='restored', key=key)),
                                     {'length': 32, 'step': 0})
                pipe.sessions.items['restored'].length = 32
                ids = torch.tensor([[35, 36, 37, 38, 39]])
                out_a = pipe.prepare('source', ids, speculative=True, return_logits=True)
                out_b = pipe.prepare('restored', ids, speculative=True, return_logits=True)
                self.assertTrue(torch.equal(out_a['logits'], out_b['logits']))
                pipe.commit('source', accepted); pipe.commit('restored', accepted)
                for stage in stages:
                    a, b = stage.states['source'], stage.states['restored']
                    self.assertEqual(a.length, b.length)
                    for layer in a.win:
                        self.assertTrue(torch.equal(a.win[layer], b.win[layer]))
                    if stage.final:
                        self.assertTrue(torch.equal(stage.carry['source'], stage.carry['restored']))
                pipe.release('source'); pipe.release('restored')
            finally:
                pipe.close()

    def test_cancel_drains_native_work(self):
        stages, pipe = self.stages()
        try:
            pipe.open('cancel')
            decisions = iter((False, True))
            _, count = prefill_pipeline(pipe, 'cancel', [torch.tensor([[1, 2, 3]])] * 2,
                cancelled=lambda: next(decisions), state_only=True, prefetch_ahead=True,
                hid_rows=self.cfg.sliding_window, return_completed=True)
            self.assertEqual(count, 1)
            self.assertEqual(pipe.sessions.items['cancel'].length, 3)
            self.assertTrue(all(stage.sessions.items['cancel'].length == 3 for stage in stages))
            pipe.release('cancel')
        finally:
            pipe.close()

    def test_estimate_is_allowed_over_authenticated_worker(self):
        from tests.test_multinode_cpu import worker
        from tools.multinode.worker import StageClient
        stages, _ = self.stages()
        with worker(stages[0]) as (port, token, _):
            client = StageClient('127.0.0.1', port, token)
            try:
                client.call('open', {'session': 'net'})
                result = client.call('prefix_estimate', {'session': 'net', 'key': 'b' * 32})
                self.assertGreaterEqual(result['upper_bound_bytes'], 0)
                client.call('release', {'session': 'net'})
            finally:
                client.close()

    def test_idle_release_requires_empty_stage_and_tolerates_telemetry_failure(self):
        from unittest.mock import patch
        stages, pipe = self.stages()
        stage = stages[0]
        pipe.open('busy')
        self.assertEqual(stage.execute('idle_release', {}), {'released': False, 'reason': 'busy'})
        pipe.release('busy')
        original_device = stage.model.device
        stage.model.device = torch.device('cuda:0')
        try:
            with patch('torch.cuda.synchronize') as sync, patch('torch.cuda.empty_cache') as empty, \
                 patch('torch.cuda.memory_allocated', side_effect=OSError('telemetry')):
                result = stage.execute('idle_release', {})
                self.assertTrue(result['released'])
                self.assertFalse(stage.poisoned)
                sync.assert_called_once(); empty.assert_called_once()
        finally:
            stage.model.device = original_device
            pipe.close()

    def test_idle_release_is_opt_in_and_waits_for_last_session(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from tools.multinode.generation import GreedyEngine
        class Client:
            manifest = {'max_rows': 32, 'mode': 'cached-ds41-spec-v1'}
            def __init__(self): self.calls = []
            def call(self, op, args):
                self.calls.append(op)
                return {'released': False, 'reason': 'not_cuda'}
        class Pipe:
            depth = 4
            def __init__(self):
                self.clients = [Client(), Client()]
                self.sessions = SimpleNamespace(items={'a': object(), 'b': object()})
            def release(self, name): self.sessions.items.pop(name)
        for enabled in ('0', '1'):
            with patch.dict('os.environ', {'DS41_CLUSTER_IDLE_RELEASE': enabled}):
                pipe = Pipe(); engine = GreedyEngine(pipe, vocab_size=512, prefill_chunk=16)
                a = SimpleNamespace(name='a'); b = SimpleNamespace(name='b')
                engine.active = {'a': a, 'b': b}
                engine._finish(a, 'length')
                self.assertEqual(pipe.clients[0].calls, [])
                engine._finish(b, 'length')
                self.assertEqual(pipe.clients[0].calls, ['idle_release'] if enabled == '1' else [])
                self.assertEqual(pipe.clients[1].calls, pipe.clients[0].calls)


if __name__ == '__main__':
    unittest.main()
