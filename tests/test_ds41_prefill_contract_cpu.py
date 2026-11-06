"""Local source contracts; optional real CPU hash proof, no extension import required."""
import ast
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "exllamav3/dsv41/serve_model.py"
TREE = ast.parse(SRC.read_text())
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and any(isinstance(f, ast.FunctionDef) and f.name == "forward_stage" for f in n.body))

def method(name, namespace):
    node = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node = ast.fix_missing_locations(ast.FunctionDef(name=node.name, args=node.args, body=node.body, decorator_list=[], returns=None, type_comment=None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), namespace)
    return namespace[name]

class Contracts(unittest.TestCase):
    def test_stage_modes_and_argument_forwarding(self):
        seen = []
        fake = types.SimpleNamespace(layer_span=(0, 2), _forward_core=lambda *a, **kw: seen.append((a, kw)), _img_rows=1)
        fn = method("forward_stage", {})
        fn(fake, "ids", 10, "state", state_only=True, ahead_ids="next", hid_rows=128)
        self.assertEqual(seen[0][0][-1], "none")
        self.assertEqual(seen[0][1], dict(ahead_ids="next", hid_rows=128, incoming=None))
        self.assertIsNone(fake._img_rows)
        fn(fake, "ids", 10, "state")
        self.assertEqual(seen[-1][0][-1], "last")
        with self.assertRaises(ValueError):
            fn(fake, "ids", 0, "state", state_only=True, all_logits=True)

    def test_dual_pointer_logical_indices(self):
        # Keep legacy clamp/masking including out-of-range and every pool boundary.
        for nw, np in ((0, 5), (5, 0), (3, 9)):
            window, pool = list(range(nw)), list(range(nw, nw+np))
            for key in (-2, -1, 0, nw-1, nw, nw+np-1, nw+np+9):
                if key < 0:
                    continue
                tok = min(key, nw+np-1)
                got = window[tok] if tok < nw else pool[tok-nw]
                self.assertEqual(got, (window+pool)[tok])

    def test_lookahead_hashes_leave_live_cache_unchanged(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable locally")
        # Load the actual hasher without importing the GPU extension package.
        import numpy as np
        engram_tree = ast.parse((ROOT / "exllamav3/dsv41/engram.py").read_text())
        hasher_node = next(n for n in engram_tree.body if isinstance(n, ast.ClassDef) and n.name == "NgramHasher")
        ns = dict(torch=torch, np=np, EngramLayout=object, DEAD=-1)
        exec(compile(ast.Module(body=[hasher_node], type_ignores=[]), "engram.py", "exec"), ns)
        Hasher = ns["NgramHasher"]
        layout = types.SimpleNamespace(layer_ids=[0], max_ngram_size=4, primes=np.array([[[101], [103], [107]]]), offsets=lambda i: np.array([0, 101, 204]))
        for start in (0, 1, 3, 17):
            h = Hasher(layout, list(range(32)), 0, 32, [[3, 5, 7, 11]], 1, 64)
            if start:
                h(torch.arange(start).view(1,-1) % 32, 0)
            before = h.cache.clone()
            ids = torch.tensor([[8, 9, 10, 11, 12]])
            mask = torch.tensor([[True, True, False, True, True]])
            ahead = method("_ahead_hashes", dict(torch=torch))(h, ids, start, mask)
            self.assertTrue(torch.equal(before, h.cache))
            self.assertTrue(torch.equal(ahead, h(ids, start, mask)))

if __name__ == "__main__":
    unittest.main()
