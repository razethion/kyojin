"""Request lifecycle contracts, including actual authenticated TCP workers."""
import threading
import unittest
from contextlib import ExitStack

import torch

from tools.multinode.generation import GreedyEngine
from tools.multinode.session import Sessions
from tools.multinode.speculative import MODE, SpeculativePipeline


class Client:
    def __init__(self, begin, end):
        self.manifest = dict(protocol=1, mode=MODE, model_id='weights', source_id='source',
                             total_modules=4, begin=begin, end=end, context=256,
                             sessions=2, cache_bits=8, max_history=7, max_rows=32)
        self.state, self.events = Sessions(256, 2), []
        self.closed, self.fail_commit, self.hook = False, False, None
        self.reject_index = None

    def call(self, op, args):
        name = args['session']
        self.events.append((op, name))
        if self.hook:
            self.hook(op)
        if op == 'open':
            return self.state.open(name)
        if op == 'draft':
            current = int(args['input_ids'][0, 0])
            tokens = torch.arange(current + 1, current + 1 + args['depth']).view(1, -1) % 100
            if self.reject_index is not None and self.reject_index < tokens.shape[1]:
                tokens[0, self.reject_index] = (tokens[0, self.reject_index] + 5) % 100
            return {'tokens': tokens}
        if op == 'prepare':
            self.state.begin(name, args['step'], args['past'], args['input_ids'].shape[1])
            self.state.mark_ready(name)
            return {'tokens': (args['input_ids'] + 1) % 100}
        if op == 'commit':
            if self.fail_commit:
                raise OSError('injected uncertain commit')
            return self.state.commit(name, args['step'], args['accepted'])
        if op == 'release':
            return self.state.release(name)
        raise ValueError(op)

    execute = call

    def close(self):
        self.closed = True


def engine(clients=None):
    clients = clients or [Client(0, 2), Client(2, 4)]
    return GreedyEngine(SpeculativePipeline(clients), vocab_size=100, prefill_chunk=32), clients


def finish(request):
    return [t for event in request for t in event['token_ids']]


class GenerationTests(unittest.TestCase):
    def test_interleaving_cancel_reuse_over_tcp(self):
        from test_multinode_cpu import worker
        from tools.multinode.worker import StageClient
        stages = [Client(0, 2), Client(2, 4)]
        with ExitStack() as stack:
            clients = []
            for stage in stages:
                port, token, _ = stack.enter_context(worker(stage))
                clients.append(StageClient('127.0.0.1', port, token))
            e, _ = engine(clients)
            try:
                a = e.start(torch.tensor([[4, 5]]), max_new_tokens=11)
                b = e.start(torch.tensor([[19, 20]]), max_new_tokens=9)
                outputs = {a.name: [], b.name: []}
                while not (a.done and b.done):
                    for r in (a, b):
                        if not r.done:
                            outputs[r.name].extend(r.step()['token_ids'])
                self.assertEqual(outputs[a.name], list(range(6, 17)))
                self.assertEqual(outputs[b.name], list(range(21, 30)))
                cancelled = e.start(torch.tensor([[30, 31]]), max_new_tokens=4)
                cancelled.step(); cancelled.cancel(); cancelled.step()
                self.assertEqual(cancelled.finish_reason, 'cancelled')
                replacement = e.start(torch.tensor([[40, 41]]), max_new_tokens=4)
                self.assertEqual(finish(replacement), [42, 43, 44, 45])
                self.assertTrue(all(not s.state.items for s in stages))
            finally:
                e.close()

    def test_eos_inside_verify_block_commits_only_prefix(self):
        e, stages = engine()
        r = e.start(torch.tensor([[4, 5]]), max_new_tokens=12, stop_ids=[8])
        self.assertEqual(finish(r), [6, 7])
        self.assertEqual((r.generated, r.finish_reason), (3, 'stop'))
        self.assertTrue(all(not s.state.items for s in stages))
        e.close()

    def test_rejected_suffix_never_leaks_into_output(self):
        for index in range(7):
            e, stages = engine(); stages[-1].reject_index = index
            r = e.start(torch.tensor([[4, 5]]), max_new_tokens=17)
            self.assertEqual(finish(r), list(range(6, 23)))
            self.assertTrue(all(not s.state.items for s in stages)); e.close()

    def test_cancel_at_each_transaction_phase_never_releases_pending(self):
        for phase in ('prepare', 'draft', 'commit'):
            for prefill in (False, True):
                if phase == 'draft' and prefill:
                    continue
                with self.subTest(phase=phase, prefill=prefill):
                    e, stages = engine()
                    r = e.start(torch.tensor([[4, 5]]), max_new_tokens=20)
                    if not prefill:
                        r.step()
                    stages[-1].hook = lambda op: r.cancel() if op == phase else None
                    event = r.step()
                    self.assertEqual(event['token_ids'], [])
                    self.assertEqual(r.finish_reason, 'cancelled')
                    self.assertTrue(all(not s.state.items for s in stages))
                    self.assertFalse(e.pipeline.failed)
                    e.close()

    def test_foreground_cancel_during_blocked_round_waits_for_commit(self):
        e, stages = engine()
        r = e.start(torch.tensor([[4, 5]]), max_new_tokens=20); r.step()
        entered, resume, results, errors = threading.Event(), threading.Event(), [], []
        def hook(op):
            if op == 'prepare':
                entered.set()
                if not resume.wait(5):
                    raise TimeoutError('test barrier expired')
        stages[-1].hook = hook
        def run():
            try: results.append(r.step())
            except BaseException as error: errors.append(error)
        thread = threading.Thread(target=run); thread.start()
        try:
            self.assertTrue(entered.wait(5)); r.cancel()
            self.assertTrue(any(s.state.items[r.name].pending for s in stages))
        finally:
            resume.set(); thread.join(5)
        self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
        self.assertEqual(results[0]['token_ids'], [])
        self.assertTrue(all(not s.state.items for s in stages)); e.close()

    def test_invalid_input_and_full_capacity_do_not_poison_live_requests(self):
        e, _ = engine()
        for ids, budget in ((torch.tensor([[100]]), 2), (torch.tensor([[1]]), True),
                            (torch.tensor([[1, 2]]), 255), (torch.tensor([[1.]]), 2)):
            with self.assertRaises(ValueError): e.start(ids, max_new_tokens=budget)
        a = e.start(torch.tensor([[4, 5]]), max_new_tokens=2)
        b = e.start(torch.tensor([[9, 10]]), max_new_tokens=2)
        with self.assertRaises(ValueError): e.start(torch.tensor([[1]]), max_new_tokens=2)
        self.assertFalse(e.pipeline.failed)
        self.assertEqual(finish(a), [6, 7]); self.assertEqual(finish(b), [11, 12]); e.close()

    def test_chunked_prefill_and_near_context_limit(self):
        e, stages = engine()
        ids = torch.zeros((1, 252), dtype=torch.int64); ids[0, -1] = 5
        r = e.start(ids, max_new_tokens=4)
        events = list(r)
        self.assertEqual(sum(event['phase'] == 'prefill' for event in events), 8)
        self.assertEqual([t for event in events for t in event['token_ids']], [6, 7, 8, 9])
        self.assertTrue(all(not s.state.items for s in stages)); e.close()

    def test_single_token_prompt_and_iterator_close(self):
        e, stages = engine()
        r = e.start(torch.tensor([[5]]), max_new_tokens=12)
        stream = iter(r)
        self.assertEqual(next(stream)['token_ids'], [6])
        stream.close()
        self.assertEqual(r.finish_reason, 'cancelled')
        self.assertTrue(all(not s.state.items for s in stages))
        replacement = e.start(torch.tensor([[2, 3]]), max_new_tokens=3)
        self.assertEqual(finish(replacement), [4, 5, 6]); e.close()

    def test_uncertain_commit_fails_all_requests_without_recycling(self):
        e, stages = engine()
        a = e.start(torch.tensor([[4, 5]]), max_new_tokens=20)
        b = e.start(torch.tensor([[9, 10]]), max_new_tokens=20)
        a.step(); b.step(); stages[-1].fail_commit = True
        with self.assertRaises(OSError): a.step()
        self.assertTrue(e.pipeline.failed and e.closed)
        self.assertEqual((a.finish_reason, b.finish_reason), ('error', 'error'))
        self.assertTrue(all(s.closed for s in stages))
        with self.assertRaises(RuntimeError): e.start(torch.tensor([[1]]), max_new_tokens=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
