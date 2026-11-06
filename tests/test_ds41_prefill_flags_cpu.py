"""Local opt-in production flags; no GPU initialization."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SRC=Path(__file__).resolve().parents[1]/"exllamav3/dsv41/serve_model.py"
TREE=ast.parse(SRC.read_text())

def helper(name,namespace):
    node=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name==name)
    node=ast.FunctionDef(name=node.name,args=node.args,body=node.body,decorator_list=[],returns=None,type_comment=None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SRC),"exec"),namespace)
    return namespace[name]

class PrefillFlags(unittest.TestCase):
    def test_dual_default_and_concat_optout_keep_wrapper_abi(self):
        calls=[];cats=[]
        def cat(values,dim):cats.append((values,dim));return "concatenated"
        fn=helper("_sparse_prefill",dict(os=os,torch=SimpleNamespace(cat=cat),
                  _pf=SimpleNamespace(sparse_attention=lambda *a:calls.append(a))))
        pool=SimpleNamespace(size=lambda dim:17)
        with patch.dict(os.environ,{},clear=True):fn("q","window","sink","idx",0.125,pool)
        self.assertEqual(calls[-1],("q","window","sink","idx",0.125,pool))
        self.assertFalse(cats)
        with patch.dict(os.environ,{"DSV41_PF_DUAL_POOL":"0"}):fn("q","window","sink","idx",0.125,pool)
        self.assertEqual(cats[-1],(("window",pool),1))
        self.assertEqual(calls[-1],("q","concatenated","sink","idx",0.125,None))
        with patch.dict(os.environ,{"DSV41_PF_DUAL_POOL":"0"}):fn("q","window","sink","idx",0.125,None)
        self.assertEqual(calls[-1],("q","window","sink","idx",0.125,None))
        self.assertEqual(len(cats),1)

    def test_candidate_order_is_optin_at_source_only(self):
        method=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name=="_indexer_hip")
        flag=next(n for n in ast.walk(method) if isinstance(n,ast.If) and "DSV41_CANONICAL_CANDIDATE_ORDER" in ast.unparse(n.test))
        self.assertIn("'0'",ast.unparse(flag.test))
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="_canonical_candidate_blocks" for n in ast.walk(method)),1)

    def test_real_tensor_order_preserves_membership_and_masks(self):
        try:import torch
        except ImportError:self.skipTest("torch unavailable")
        fn=helper("_canonical_candidate_blocks",dict(torch=torch))
        blocks=torch.tensor([[65,231,2,-1],[231,65,-1,2]],dtype=torch.int32)
        out=fn(blocks)
        self.assertEqual(out.tolist(),[[2,65,231,-1],[2,65,231,-1]])
        self.assertTrue(torch.equal(blocks.sort(-1).values,out.sort(-1).values))
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.dtype,blocks.dtype)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
