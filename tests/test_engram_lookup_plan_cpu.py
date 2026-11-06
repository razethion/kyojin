import unittest
from unittest.mock import patch
import torch
import importlib.util
import sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('lookup_engram',Path(__file__).resolve().parents[1]/'exllamav3/dsv41/engram.py')
eg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=eg;spec.loader.exec_module(eg)
class Store:
    fmt='fp8'
    def __init__(self):
        self.raw=torch.arange(16*264).remainder(120).to(torch.uint8).reshape(16,264)
        self.raw[:,256:]=127
    def fetch(self,layer,rows):return self.raw[rows]

class LookupPlanTests(unittest.TestCase):
    def test_canonical_one_position_no_unique_and_reuse_exact(self):
        layout=eg.EngramLayout.build([1],[16],2,2,2,256,primes=[[[5,11]]])
        hashes=torch.tensor([[[2,8]]]);h=torch.randn(1,1,2,4).half()
        weight=torch.randn(512,12).half();wkv=lambda x:x@weight
        expected=eg.engram_inject(h,hashes,Store(),0,wkv,torch.ones(2,4),2,4,1e-6)
        store=eg.PrefetchRowStore(Store())
        try:
            with patch.object(torch,'unique',side_effect=AssertionError('duplicate unique')):
                plan=eg.EngramLookupPlan.build(hashes,0,layout=layout)
                store.prefetch_plan(plan)
                actual=eg.engram_inject(h,hashes,store,0,wkv,torch.ones(2,4),2,4,1e-6,lookup_plan=plan)
            self.assertTrue(torch.equal(expected,actual));self.assertEqual(store.hits,1)
        finally:store.close()
    def test_arbitrary_duplicate_fallback_snapshot(self):
        ids=torch.tensor([[[7,2,7]]]);plan=eg.EngramLookupPlan.build(ids,0)
        ids.zero_()
        self.assertEqual(plan.rows.tolist(),[2,7]);self.assertEqual(plan.inverse.tolist(),[1,0,1])
    def test_geometry_and_layer_guard(self):
        plan=eg.EngramLookupPlan.build(torch.tensor([[[2]]]),0)
        with self.assertRaisesRegex(ValueError,'geometry/layer'):
            eg.engram_inject(torch.zeros(1,1,1,1),torch.tensor([[[2]]]),Store(),1,None,None,1,1,1e-6,lookup_plan=plan)

if __name__=='__main__':unittest.main()
