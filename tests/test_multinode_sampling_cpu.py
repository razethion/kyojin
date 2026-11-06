import unittest
import torch
from tools.multinode.sampling import make_sampler, sampled_prefix, RepetitionGuard, validate_sampling
from tools.multinode.generation import GreedyEngine
from tools.multinode.prefill_core import PrefillEngine
from tools.multinode.speculative import SpeculativePipeline, DS41_MODE
from test_multinode_generation_cpu import Client, finish


class SampleClient(Client):
    def __init__(self, begin, end):
        super().__init__(begin, end)
        self.manifest['mode'] = DS41_MODE
        self.table = torch.randn(100, 100, generator=torch.Generator().manual_seed(42)) * 2

    def call(self, op, args):
        result = super().call(op, args)
        if op == 'prepare':
            logits = self.table[args['input_ids']]
            if not args['speculative']:
                logits = logits[:, -1:]
            result = {'tokens': logits.argmax(-1)}
            if args['return_logits']:
                result['logits'] = logits
        return result


def setup(core=GreedyEngine):
    clients = [SampleClient(0, 2), SampleClient(2, 4)]
    return core(SpeculativePipeline(clients, mode=DS41_MODE), vocab_size=100,
                prefill_chunk=32, adaptive_drafting=False), clients


class SamplingTests(unittest.TestCase):
    def test_plain_and_speculative_exact_rng_and_budget(self):
        for seed in (0, 1, 829):
            for top_p in (1.0, .95, .4):
                e, clients = setup()
                try:
                    args = dict(max_new_tokens=27, temperature=1., top_p=top_p, seed=seed)
                    ids = torch.tensor([[3, 5, 7]])
                    plain = finish(e.start(ids, speculative=False, **args))
                    spec = finish(e.start(ids, **args))
                    self.assertEqual(spec, plain)
                    self.assertEqual(len(spec), 27)
                    sampler = make_sampler(1., top_p, seed)
                    expected = []; token = 7
                    for _ in range(27):
                        token = sampler.sample(clients[-1].table[token]); expected.append(token)
                    self.assertEqual(spec, expected)
                finally:
                    e.close()

    def test_every_acceptance_boundary_and_no_unused_rng(self):
        logits = torch.randn(5, 100, generator=torch.Generator().manual_seed(1))
        for accepted in range(5):
            for remaining in (1, 2, 5):
                ref = make_sampler(1., .95, 123)
                draws = [ref.sample(row) for row in logits]
                draft = draws[:4]
                if accepted < 4:
                    draft[accepted] = (draft[accepted] + 1) % 100
                sampler = make_sampler(1., .95, 123)
                got = sampled_prefix(draft, logits, remaining, sampler)
                n = min(accepted + 1, remaining, 5)
                self.assertEqual(got['emitted'], draws[:n])
                reference = make_sampler(1., .95, 123)
                for row in logits[:n]: reference.sample(row)
                self.assertTrue(torch.equal(sampler.generator.get_state(), reference.generator.get_state()))

    def test_two_session_rng_isolation_and_overlap(self):
        e, _ = setup(PrefillEngine)
        e.paired_draft_overlap = True
        try:
            args = dict(max_new_tokens=19, temperature=1., top_p=.95)
            ids = torch.tensor([[3, 5, 7]])
            expected = [finish(e.start(ids, seed=s, **args)) for s in (7, 19)]
            reqs = [e.start(ids, seed=s, **args) for s in (7, 19)]
            outputs = [[], []]
            while not all(r.done for r in reqs):
                for out, event in zip(outputs, e.step_many(reqs)):
                    out.extend(event['token_ids'])
            self.assertEqual(outputs, expected)
            self.assertFalse(e.active)
        finally: e.close()

    def test_eos_sampling_consumes_only_through_eos(self):
        logits = torch.zeros(4, 100)
        ref = make_sampler(1., 1., 2)
        first = ref.sample(logits[0])
        sampler = make_sampler(1., 1., 2)
        got = sampled_prefix([first, 8, 9], logits, 4, sampler, [first])
        self.assertEqual(got['emitted'], [first])
        self.assertTrue(torch.equal(ref.generator.get_state(), sampler.generator.get_state()))

    def test_guard_extreme_periodic_only(self):
        for pattern in ([4], [4, 12], list(range(16))):
            guard = RepetitionGuard()
            for i in range(255): self.assertFalse(guard.push(pattern[i % len(pattern)]))
            self.assertTrue(guard.push(pattern[255 % len(pattern)]))
        guard = RepetitionGuard()
        for i in range(2000): self.assertFalse(guard.push(i % 17))

    def test_guard_releases_committed_session(self):
        from tools.multinode.session import Sessions
        clients = [SampleClient(0, 2), SampleClient(2, 4)]
        for c in clients:
            c.manifest['context'] = 1024
            c.state = Sessions(1024, 2)
            c.table.zero_(); c.table[:, 1] = 100
        e = GreedyEngine(SpeculativePipeline(clients, mode=DS41_MODE), vocab_size=100,
                        prefill_chunk=32, adaptive_drafting=False)
        try:
            r = e.start(torch.tensor([[3, 5, 7]]), max_new_tokens=512, repetition_guard=True)
            self.assertEqual(finish(r), [1] * 256)
            self.assertEqual(r.finish_reason, 'repetition')
            self.assertFalse(e.active)
            self.assertFalse(e.pipeline.sessions.items)
            self.assertEqual(finish(e.start(torch.tensor([[3, 5]]), max_new_tokens=2)), [1, 1])
        finally: e.close()

    def test_arguments(self):
        for values in ((float('nan'), 1, 0), (1, 0, 0), (1, 1, -1), (True, 1, 0), (1, 1, True)):
            with self.assertRaises(ValueError): validate_sampling(*values)

    def test_native_tiny_model_sampled_mtp_matches_plain(self):
        from test_multinode_ds41_cpu import world, Local, DeepSeekStage
        torch.set_num_threads(2)
        cfg, full, a, b = world()
        stages = [DeepSeekStage(m, model_id='tiny', source_id='same', context=256,
                               sessions=2, depth=4, max_rows=32) for m in (a, b)]
        e = GreedyEngine(SpeculativePipeline([Local(s) for s in stages], mode=DS41_MODE),
                        vocab_size=cfg.vocab_size, prefill_chunk=16, adaptive_drafting=False)
        try:
            with torch.inference_mode():
                for seed in (0, 17):
                    ids = torch.tensor([[3, 7, 11, 9, 15, 21]])
                    args = dict(max_new_tokens=12, seed=seed, temperature=1., top_p=.95)
                    plain = finish(e.start(ids, speculative=False, **args))
                    spec = finish(e.start(ids, **args))
                    self.assertEqual(spec, plain)
        finally:
            e.close()


class SampleHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampled_http_seed_plain_and_stream(self):
        import json
        from aiohttp.test_utils import TestClient, TestServer
        from tools.multinode.serve_ds41 import DeepSeekClusterEngine, create_app
        from test_multinode_glm_http_cpu import Tokenizer
        core, _ = setup()
        client = TestClient(TestServer(create_app(DeepSeekClusterEngine(core, Tokenizer(), []), 'sample')))
        await client.start_server()
        try:
            body = dict(model='sample', messages=[dict(role='user', content='hello')],
                        thinking=False, seed=42, max_tokens=19)
            async def run(**kw):
                response = await client.post('/v1/chat/completions', json=body | kw)
                self.assertEqual(response.status, 200, await response.text())
                return await response.json()
            sampled = await run()
            plain = await run(speculative=False)
            self.assertEqual(sampled['choices'], plain['choices'])
            other = await run(seed=43)
            self.assertNotEqual(sampled['choices'], other['choices'])
            response = await client.post('/v1/chat/completions', json=body | dict(stream=True))
            raw = await response.text()
            events = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith('data: {')]
            text = ''.join(v['choices'][0].get('delta', {}).get('content', '') for v in events if v.get('choices'))
            self.assertEqual(text, sampled['choices'][0]['message']['content'])
            self.assertFalse(core.active)
        finally: await client.close()


if __name__ == '__main__': unittest.main()
