import unittest
from types import SimpleNamespace

from tools.multinode.prefill_pipeline import prefill_pipeline


class Ids:
    def __init__(self, n):
        self.shape = (1, n)


class Sessions:
    def __init__(self):
        self.items = {'s': SimpleNamespace(step=0, length=0)}
        self.pending = None

    def begin(self, name, step, past, rows):
        s = self.items[name]
        assert (s.step, s.length) == (step, past)
        self.pending = rows

    def mark_ready(self, name):
        assert self.pending is not None

    def commit(self, name, step, accepted):
        s = self.items[name]
        assert self.pending == accepted and s.step == step
        s.length += accepted
        s.step += 1
        self.pending = None


class Client:
    def __init__(self):
        self.step = self.length = 0
        self.pending = None
        self.prepares = []

    def call(self, op, args):
        if op == 'prepare':
            self.prepares.append(dict(args))
            assert (args['step'], args['past']) == (self.step, self.length)
            self.pending = args['input_ids'].shape[1]
            return 'activation'
        assert args['step'] == self.step and args['accepted'] == self.pending
        self.length += self.pending
        self.step += 1
        self.pending = None
        return {'length': self.length, 'step': self.step}


class PrefillPipelineTests(unittest.TestCase):
    def pipe(self):
        return SimpleNamespace(sessions=Sessions(), clients=[Client(), Client()],
                               close=lambda: self.fail('unexpected close'))

    def test_sixteen_chunks_and_short_tail_remain_paired(self):
        pipe = self.pipe()
        output, count = prefill_pipeline(pipe, 's', [Ids(256) for _ in range(15)] + [Ids(7)], return_completed=True)
        self.assertEqual((output, count), ('activation', 16))
        for state in [pipe.sessions.items['s'], *pipe.clients]:
            self.assertEqual((state.step, state.length), (16, 3847))

    def test_cancel_drains_already_submitted_chunk(self):
        pipe = self.pipe()
        calls = iter([False, False, True])
        output, count = prefill_pipeline(pipe, 's', [Ids(256)] * 4,
                                         cancelled=lambda: next(calls), return_completed=True)
        self.assertEqual((output, count), ('activation', 2))
        for state in [pipe.sessions.items['s'], *pipe.clients]:
            self.assertEqual((state.step, state.length), (2, 512))

    def test_native_options_only_skip_interior_logits(self):
        pipe = self.pipe()
        chunks = [Ids(256), Ids(17)]
        self.assertEqual(prefill_pipeline(pipe, 's', chunks, state_only=True,
                                         prefetch_ahead=True, hid_rows=128), 'activation')
        for client in pipe.clients:
            self.assertTrue(client.prepares[0]['state_only'])
            self.assertFalse(client.prepares[1]['state_only'])
            self.assertIs(client.prepares[0]['ahead_ids'], chunks[1] if client is pipe.clients[0] else None)
            self.assertIsNone(client.prepares[1]['ahead_ids'])
            self.assertEqual(client.prepares[0]['hid_rows'], 128)

    def test_cancel_before_first_chunk(self):
        pipe = self.pipe()
        self.assertEqual(prefill_pipeline(pipe, 's', [Ids(256)], cancelled=lambda: True, return_completed=True), (None, 0))
        self.assertEqual(pipe.sessions.items['s'].length, 0)


if __name__ == '__main__':
    unittest.main()
