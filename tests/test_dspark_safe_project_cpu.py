"""Actual DSpark main projection helper: CPU overflow/math gates, no extension import."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]/'exllamav3/dsv41'
spec=importlib.util.spec_from_file_location('safe_project_ops',ROOT/'ops.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
tree=ast.parse((ROOT/'dspark.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DSparkHeads')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_project_main')
ns=dict(torch=torch,os=os,ops=ops)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'dspark.py','exec'),ns)

class Projection:
    def __init__(self,w):self.w=w;self.inputs=[];self.lin=SimpleNamespace(inner=SimpleNamespace(bias=None))
    def __call__(self,x):
        self.inputs.append(x.clone())
        return F.linear(x.half(),self.w).float()

class SafeProjection(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(8077)
        proj=Projection((torch.randn(64,128)*.25).half())
        heads=SimpleNamespace(main_proj=proj,main_norm=torch.rand(64)+.5,
                              m=SimpleNamespace(cfg=SimpleNamespace(rms_norm_eps=1e-6)))
        return heads
    def call(self,h,x,flag):
        with patch.dict(os.environ,DSV41_MTP_SAFE_PROJECT=flag):return ns['_project_main'](h,x)
    def test_default_and_ordinary_rows_bitwise(self):
        h=self.fixture();x=torch.randn(1,127,128)*20
        expected=ops.rmsnorm(h.main_proj(x),h.main_norm,1e-6)
        self.assertTrue(torch.equal(self.call(h,x,'0'),expected))
        self.assertTrue(torch.equal(self.call(h,x,'1'),expected))
        self.assertTrue(torch.equal(h.main_proj.inputs[-1],x))
    def test_large_rows_finite_and_fp32_reference(self):
        h=self.fixture();x=torch.randn(1,127,128)
        x=x/x.abs().amax(-1,keepdim=True)*275842.25
        x[:,0]=x[:,0]/10000  # mixed ordinary/overflow rows preserve row-local behavior
        expected=ops.rmsnorm(F.linear(x,h.main_proj.w.float()),h.main_norm,1e-6)
        legacy=self.call(h,x,'0');self.assertFalse(bool(legacy.isfinite().all()))
        actual=self.call(h,x,'1');self.assertTrue(bool(actual.isfinite().all()))
        torch.testing.assert_close(actual,expected,atol=.002,rtol=.002)
        scaled=h.main_proj.inputs[-1]
        self.assertLessEqual(float(scaled.abs().max()),128)
        self.assertTrue(torch.equal(scaled[:,0],x[:,0]))
        # Each complete row has one positive power-of-two scale, no clipping.
        scale=scaled[:,:,0:1]/x[:,:,0:1]
        torch.testing.assert_close(scaled,x*scale,atol=0,rtol=0)
        self.assertTrue(torch.equal(torch.log2(scale),torch.log2(scale).round()))
        # Unnormalized 128-point input Hadamard bound is at most 128*128.
        self.assertLess(128*128,torch.finfo(torch.float16).max)
    def test_correct_scaled_epsilon_and_bias_guard(self):
        h=self.fixture();h.m.cfg.rms_norm_eps=1000
        x=torch.randn(1,4,128)*10000
        actual=self.call(h,x,'1')
        projected=h.main_proj(h.main_proj.inputs[-1])
        scale=h.main_proj.inputs[-1][:,:,0:1]/x[:,:,0:1]
        expected=ops.rmsnorm(projected,h.main_norm,1000*scale.square())
        self.assertTrue(torch.equal(actual,expected))
        h.main_proj.lin.inner.bias=torch.zeros(64)
        with self.assertRaisesRegex(ValueError,'bias-free'):self.call(h,x,'1')

if __name__=='__main__':unittest.main()
