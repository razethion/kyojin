import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1] / 'exllamav3' / 'dsv41'
pkg = types.ModuleType('draft_test_pkg')
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
ops = types.ModuleType('draft_test_pkg.ops')
ops.hc_pre = lambda h, pre: h[:, :, 0, :]
ops.rmsnorm = lambda x, weight, eps: x
sys.modules[ops.__name__] = ops
sys.modules['draft_test_pkg.quant'] = types.ModuleType('draft_test_pkg.quant')
linear = types.ModuleType('draft_test_pkg.linear')
linear.DenseLin = object
sys.modules[linear.__name__] = linear

def load(name):
    spec = importlib.util.spec_from_file_location('draft_test_pkg.' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

DSparkHeads = load('dspark').DSparkHeads
PortDrafter = load('spec').PortDrafter

class DraftHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        cfg = types.SimpleNamespace(dspark_block_size=5, dspark_noise_token_id=0, hc_mult=1,
                                    num_hidden_layers=0, rms_norm_eps=1e-5, sliding_window=16)
        self.widths = []
        def head(x):
            self.widths.append(x.shape[1])
            return x @ self.weight
        self.weight = torch.randn(4, 11)
        model = types.SimpleNamespace(cfg=cfg, device=torch.device('cpu'), embed_w=torch.randn(11,4), head_logits=head)
        self.heads = DSparkHeads.__new__(DSparkHeads)
        self.heads.m = model
        self.heads.n_stages = 1
        self.heads.main_proj = lambda x: x
        self.heads.main_norm = self.heads.norm = torch.ones(4)
        self.heads.markov_embed = torch.randn(11,3)
        self.heads.markov_head = torch.randn(11,3)
        self.heads.conf = torch.randn(1,7)
        self.heads._mh = self.heads.rhead = None
        self.trunk_widths = []
        def block(layer, h, pos, pre, main, state, taps):
            self.trunk_widths.append(h.shape[1])
            # Every output depends on all five positions.
            return h + h.mean(dim=1, keepdim=True), pre
        self.heads._block = block
        model.mtp = self.heads
        self.spec = types.SimpleNamespace(model=model, st=types.SimpleNamespace(win=[torch.zeros(1,16,4)]))
        self.hidden = torch.randn(1,4)

    def draft(self, want, enabled, max_depth=5):
        with patch.dict(os.environ, {'DSV41_DRAFT_PARTIAL_HEAD': str(int(enabled))}):
            return PortDrafter(self.spec, max_depth).draft(2, self.hidden, 2, want=want)

    def test_exact_proposals_confidence_and_full_width(self):
        for want in range(1, 6):
            reference = self.draft(want, False)
            with patch('torch.nn.functional.linear', wraps=torch.nn.functional.linear) as linear_mock:
                actual = self.draft(want, True)
            self.assertEqual(actual.tokens, reference.tokens)
            torch.testing.assert_close(actual.conf, reference.conf, rtol=0, atol=0)
            self.assertEqual(linear_mock.call_count, want + 1)
        self.assertTrue(all(x == 5 for x in self.widths + self.trunk_widths))

    def test_confidence_opt_out_is_explicit_and_independent(self):
        for partial in ("0", "1"):
            for want in (3, 8):
                with patch.dict(os.environ, {"DSV41_DRAFT_PARTIAL_HEAD": partial,
                                             "DSV41_DRAFT_SKIP_CONF": "1"}):
                    ref = PortDrafter(self.spec, 10).draft(2, self.hidden, 2, want=want)
                    with patch('torch.nn.functional.linear', wraps=torch.nn.functional.linear) as call:
                        out = PortDrafter(self.spec, 10).draft(2, self.hidden, 2, want=want, want_conf=False)
                    self.assertEqual(out.tokens, ref.tokens)
                    self.assertIsNone(out.conf)
                    self.assertIsNotNone(ref.conf)
                    count = want if partial == "1" else (5 if want <= 5 else 10)
                    self.assertEqual(call.call_count, count)
        with patch.dict(os.environ, {"DSV41_DRAFT_SKIP_CONF": "0"}):
            out = PortDrafter(self.spec).draft(2, self.hidden, 2, want=3, want_conf=False)
            self.assertIsNotNone(out.conf)

    def test_legacy_and_chained(self):
        with patch('torch.nn.functional.linear', wraps=torch.nn.functional.linear) as linear_mock:
            self.draft(3, False)
            self.assertEqual(linear_mock.call_count, 6)
        for want in (6, 8, 10):
            reference = self.draft(want, False, 10)
            actual = self.draft(want, True, 10)
            self.assertEqual(actual.tokens, reference.tokens)
            torch.testing.assert_close(actual.conf, reference.conf, rtol=0, atol=0)
        with patch('torch.nn.functional.linear', wraps=torch.nn.functional.linear) as linear_mock:
            self.draft(None, True)
            self.assertEqual(linear_mock.call_count, 6)

if __name__ == '__main__':
    unittest.main()
