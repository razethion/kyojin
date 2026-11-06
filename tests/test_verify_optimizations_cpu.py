"""Dependency-free rollback state proof; optional real GPU tail proof lives separately."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class Tensor:
    device = 'cpu'
    def __init__(self, rows): self.rows = [list(r) for r in rows]
    def size(self, dim): return (len(self.rows), len(self.rows[0]))[dim]
    def clone(self): return Tensor(self.rows)
    def __mod__(self, n): return [v % n for v in self.rows[0]]
    def __getitem__(self, key):
        b, cols = key
        rows = self.rows[b]
        if isinstance(cols, int): return Tensor([[r[cols]] for r in rows])
        return Tensor([[r[c] for c in cols] for r in rows])
    def __setitem__(self, key, value):
        b, col = key
        for i, r in enumerate(self.rows[b]): r[col] = value.rows[i][0]

FAKE_TORCH = SimpleNamespace(tensor=lambda xs, **kw: xs,
                             arange=lambda a, b, **kw: Tensor([range(a,b)]))

def load_state():
    path = ROOT / 'exllamav3/dsv41/model.py'
    tree = ast.parse(path.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('DSV41State', '_BlockJournal')]
    classes[0].body = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name != '__init__']
    ns = dict(torch=FAKE_TORCH, os=os)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['DSV41State']

class StateProof(unittest.TestCase):
    def test_attention_reuses_forward_owned_position_views(self):
        path = ROOT / 'exllamav3/dsv41/serve_model.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DSV41Serving')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_attention_verify')
        ns = {'torch': SimpleNamespace(cat=lambda xs, dim: xs[0])}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'),ns)
        class X:
            def size(self, dim): return 4
            def __getitem__(self, key): return self
            def contiguous(self): return self
            def flatten(self, dim): return self
        x = X()
        seen = []
        w = SimpleNamespace(wq_a=lambda x:x, wq_b=lambda x:x, wkv=lambda x:x,
                            q_norm=None, wo_b=lambda x:x)
        fake = SimpleNamespace(lw={0:w}, _pfrms=lambda *args:x, _wo_a=lambda *args:x)
        fake._attention = lambda *args, **kw: seen.append((fake._pos32,fake._slot32)) or x
        positions = [(object(),object()) for _ in range(4)]
        # No tensor allocator exists in ns: any per-row allocation is a failure.
        ns['_attention_verify'](fake,0,x,7,None,[None]*4,row_positions=positions)
        ns['_attention_verify'](fake,0,x,7,None,[None]*4,row_positions=positions)
        for i, pair in enumerate(seen):
            self.assertIs(pair[0],positions[i%4][0])
            self.assertIs(pair[1],positions[i%4][1])

    def test_every_accepted_prefix_wrap_batch_and_compressor_reuse(self):
        State = load_state()
        for batch in (1, 2):
            for start in (0, 3, 7, 8, 15):
                for length in range(1, 9):
                    for keep in range(length + 1):
                        states = []
                        for enabled in ('0', '1'):
                            st = State()
                            st.win = {l: Tensor([[100*b + 10*l + i for i in range(8)] for b in range(batch)]) for l in (0,1)}
                            st.comp_kv_state = {0: Tensor([[i for i in range(3)] for _ in range(batch)])}
                            st.comp_score_state = {0: Tensor([[-i for i in range(3)] for _ in range(batch)])}
                            with patch.dict(os.environ, DSV41_VERIFY_RING_BATCH=enabled): st.begin_block(start,length)
                            for l in (0,1):
                                for i in range(length):
                                    p = start+i
                                    st._jr_ring(l,[p],batch)
                                    st.win[l][:batch,p%8] = Tensor([[1000+p+b] for b in range(batch)])
                                    if l == 0:
                                        st._jr_comp(0,p,p%3,batch)
                                        st.comp_kv_state[0][:batch,p%3] = Tensor([[2000+p] for _ in range(batch)])
                                        st.comp_score_state[0][:batch,p%3] = Tensor([[3000+p] for _ in range(batch)])
                            if enabled == '1': self.assertTrue(all(len(v)==1 for v in st.journal.ring.values()))
                            st.rollback(keep)
                            self.assertIsNone(st.journal)
                            for l, ring in st.win.items():
                                expected = [[100*b+10*l+i for i in range(8)] for b in range(batch)]
                                for p in range(start,start+keep):
                                    for b in range(batch): expected[b][p%8] = 1000+p+b
                                self.assertEqual(ring.rows,expected)
                            for state, base, scale in ((st.comp_kv_state,2000,1),(st.comp_score_state,3000,-1)):
                                expected = [[scale*i for i in range(3)] for _ in range(batch)]
                                for p in range(start,start+keep):
                                    for b in range(batch): expected[b][p%3]=base+p
                                self.assertEqual(state[0].rows,expected)
                            states.append(st)
                        self.assertEqual(states[0].win[0].rows,states[1].win[0].rows)

if __name__ == '__main__': unittest.main()
