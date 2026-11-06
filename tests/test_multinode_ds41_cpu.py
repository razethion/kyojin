"""Real native tiny DSV41 math, span carries, journals and DSpark lifecycle."""
import copy
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).parent / 'dsv41'))
from harness import FakeTok, tiny_config, make_state_dict, make_engram_tables
from exllamav3.dsv41.engram import build_compressed_token_map, EngramLayout, MemoryEngramStore
from exllamav3.dsv41.weights import DictWeightSource
from exllamav3.dsv41.serve_model import DSV41Serving
from tools.multinode.ds41_stage import DeepSeekStage, MODE
from tools.multinode.speculative import SpeculativePipeline
from tools.multinode.generation import GreedyEngine


def world():
    cfg = tiny_config(index_n_heads=16, dspark_block_size=5, dspark_noise_token_id=511,
                      engram_layer_ids=[1, 3], dspark_target_layer_ids=[5, 6, 7])
    tm, cfg.engram_compressed_vocab_size = build_compressed_token_map(FakeTok(cfg.vocab_size))
    layout = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size,
        cfg.engram_vocab_size, cfg.engram_n_heads, cfg.engram_head_dim)
    cfg.engram_num_embeddings = list(layout.num_embeddings)
    tables = make_engram_tables(cfg, layout.num_embeddings, 92)
    sd = make_state_dict(cfg, 91, fp8=False, fp4=False)
    def make(span=None):
        c = copy.deepcopy(cfg)
        if span and span[1] < cfg.num_hidden_layers: c.num_nextn_predict_layers = 0
        src = DictWeightSource(sd, cfg.num_hidden_layers)
        store = MemoryEngramStore([t[0] for t in tables], [t[1] for t in tables])
        return DSV41Serving(c, src, None, store, token_map=tm, dtype=torch.float32,
                            layer_span=span, device='cpu')
    return cfg, make(), make((0, 4)), make((4, 8))


class Local:
    def __init__(self, stage): self.stage, self.manifest = stage, stage.manifest
    def call(self, op, args): return self.stage.execute(op, args)
    def close(self): pass


class DeepSeekTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg, cls.full, cls.a, cls.b = world()

    def stages(self, context=256):
        stages = [DeepSeekStage(m, model_id='tiny-native-v1', source_id='same-source',
                  context=context, sessions=2, depth=4, max_rows=32) for m in (self.a, self.b)]
        return stages, SpeculativePipeline([Local(s) for s in stages], mode=MODE)

    def test_split_chunk_decode_and_rejected_suffix(self):
        full, a, b = self.full, self.a, self.b
        sf, sa, sb = [m.new_state(1, 256) for m in (full, a, b)]
        ids = torch.randint(0, 500, (1, 80), generator=torch.Generator().manual_seed(7))
        with torch.inference_mode():
            pos = 0
            for n in (7, 13, 1, 5, 1):
                x = ids[:, pos:pos+n]
                lf, hf = full.forward_stage(x, pos, sf, all_logits=True)
                carry, _ = a.forward_stage(x, pos, sa, all_logits=True)
                lb, hb = b.forward_stage(x, pos, sb, carry, all_logits=True)
                torch.testing.assert_close(lb, lf, atol=0, rtol=0)
                torch.testing.assert_close(hb, hf, atol=0, rtol=0)
                pos += n
            for keep in range(1, 6):
                # Fresh journal at each verified prefix, with a wrapped ring.
                for st in (sf, sa, sb): st.begin_block(pos, 5)
                x = ids[:, pos:pos+5]
                lf, _ = full.forward_stage(x, pos, sf, all_logits=True)
                carry, _ = a.forward_stage(x, pos, sa, all_logits=True)
                lb, _ = b.forward_stage(x, pos, sb, carry, all_logits=True)
                torch.testing.assert_close(lb, lf, atol=0, rtol=0)
                for st in (sf, sa, sb):
                    st.rollback(keep); st.length = pos + keep
                pos += keep
            for source in sa.comp_kv: self.assertLess(source, 4)
            for source in sb.comp_kv: self.assertGreaterEqual(source, 4)

    def test_transactions_dspark_two_slots_and_single_token(self):
        stages, pipe = self.stages()
        core = GreedyEngine(pipe, vocab_size=512, prefill_chunk=7, adaptive_drafting=False)
        # Both slots are live and interleaved, including a one-token prompt.
        reqs = [core.start(torch.tensor([[4, 8, 10, 12, 6, 3, 11, 9, 15, 20]]), max_new_tokens=12),
                core.start(torch.tensor([[7]]), max_new_tokens=9)]
        outputs = [[], []]
        while any(not r.done for r in reqs):
            for i, r in enumerate(reqs):
                if not r.done: outputs[i] += r.step()['token_ids']
        self.assertEqual([len(x) for x in outputs], [12, 9])
        self.assertFalse(pipe.failed)
        self.assertTrue(all(not s.sessions.items for s in stages))
        # Same-graph serial R1 is an independent acceptance-correctness oracle.
        for prompt, tokens in zip(([4,8,10,12,6,3,11,9,15,20], [7]), outputs):
            st = self.full.new_state(1, 256)
            logits, _ = self.full.forward(torch.tensor([prompt]), 0, st, chunk=7)
            expected = []
            for j in range(len(tokens)):
                token = int(logits.argmax()); expected.append(token)
                logits, _ = self.full.forward_stage(torch.tensor([[token]]), len(prompt)+j, st)
            self.assertEqual(tokens, expected)
        r = core.start(torch.tensor([[3, 8, 2]]), max_new_tokens=9)
        r.step(); r.cancel(); r.step()
        self.assertEqual(r.finish_reason, 'cancelled')
        core.close()

    def test_cache_reset_unpacked(self):
        st = self.b.new_state(1, 256)
        st.ensure(256); size = st.nbytes(); st.reset()
        self.assertEqual(st.nbytes(), size)
        self.assertEqual(st.length, 0)

    def test_native_spans_over_authenticated_tcp(self):
        from contextlib import ExitStack
        from test_multinode_cpu import worker
        from tools.multinode.worker import StageClient
        stages, _ = self.stages()
        with ExitStack() as stack:
            clients = []
            for stage in stages:
                port, token, _ = stack.enter_context(worker(stage))
                clients.append(StageClient('127.0.0.1', port, token))
            pipe = SpeculativePipeline(clients, mode=MODE)
            core = GreedyEngine(pipe, vocab_size=512, prefill_chunk=7, adaptive_drafting=False)
            try:
                req = core.start(torch.tensor([[4,7,9,3,2,11]]), max_new_tokens=12)
                output = [t for event in req for t in event['token_ids']]
                self.assertEqual(len(output), 12)
                self.assertTrue(all(c.ping() == {'alive': True} for c in clients))
                self.assertTrue(all(not s.sessions.items for s in stages))
            finally: core.close()

    def test_context_edge_does_not_overrun_fixed_draft_block(self):
        stages,pipe=self.stages()
        core=GreedyEngine(pipe,vocab_size=512,prefill_chunk=32,adaptive_drafting=False)
        prompt=torch.randint(0,500,(1,248),generator=torch.Generator().manual_seed(73))
        request=core.start(prompt,max_new_tokens=8)
        out=[t for event in request for t in event['token_ids']]
        self.assertEqual(len(out),8)
        self.assertFalse(pipe.failed)
        self.assertTrue(all(not s.sessions.items for s in stages))
        core.close()

    def test_trace_comparison_handles_growing_index_candidate_lists(self):
        stages,pipe=self.stages()
        for name in ('oracle','block'):
            pipe.open(name);pipe.prepare(name,torch.tensor([[3,4,5]]));pipe.commit(name,3)
            for c in pipe.clients:c.call('debug',dict(action='trace',session=name))
        for token in (7,8):
            pipe.prepare('oracle',torch.tensor([[token]]),return_logits=True);pipe.commit('oracle',1)
        pipe.prepare('block',torch.tensor([[7,8]]),speculative=True,return_logits=True);pipe.commit('block',2)
        for c in pipe.clients:
            report=c.call('debug',dict(action='compare',left='oracle',right='block',rows=2,traces=True))
            self.assertEqual(report['length'],5)
            self.assertIsInstance(report['differences'],list)
        for name in ('oracle','block'):pipe.release(name)
        pipe.close()


if __name__ == '__main__': unittest.main()
