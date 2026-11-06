import threading
import unittest
from unittest.mock import patch
import torch
from exllamav3.dsv41 import engram as eg


class Store:
    fmt='fp8'
    def __init__(self):
        self.raw=torch.arange(16*264).remainder(120).to(torch.uint8).reshape(16,264)
        self.raw[:,256:]=127
    def num_rows(self,layer):return 16
    def fetch(self,layer,rows):return self.raw[rows]


class PreparedPrefetchTests(unittest.TestCase):
    def test_worker_dequant_and_actual_injection_exact(self):
        torch.manual_seed(12);store=Store();ids=torch.tensor([1,2])
        hashes=torch.tensor([[[1],[2],[1]]]);h=torch.randn(1,3,2,4).half()
        weight=torch.randn(256,12).half();wkv=lambda x:x@weight
        expected=eg.engram_inject(h,hashes,store,0,wkv,torch.ones(2,4),2,4,1e-6)
        prepared=eg.PrefetchRowStore(store,prepared_cpu=True)
        calls=[];canonical=eg.dequant_rows
        def record(*args,**kwargs):
            calls.append(threading.current_thread().name)
            return canonical(*args,**kwargs)
        try:
            with patch.object(eg,'dequant_rows',side_effect=record):
                prepared.prefetch(0,ids)
                actual=eg.engram_inject(h,hashes,prepared,0,wkv,torch.ones(2,4),2,4,1e-6)
            self.assertTrue(torch.equal(actual,expected))
            self.assertEqual(prepared.prepared_hits,1)
            self.assertEqual(len(calls),1);self.assertTrue(calls[0].startswith('engram-prefetch'))
        finally:prepared.close()

    def test_bound_fallback_exact_ids_and_pending_limit(self):
        store=Store();prepared=eg.PrefetchRowStore(store,max_pending=2,prepared_cpu=True,prepared_max_bytes=0)
        try:
            ids=torch.tensor([1,2]);prepared.prefetch(0,ids);ids[0]=9
            self.assertEqual(len(prepared.pending),1)
            result=prepared.fetch_dequant(0,torch.tensor([1,2]),torch.float16)
            self.assertTrue(torch.equal(result,eg.dequant_rows(store.raw[[1,2]],torch.float16)))
            self.assertEqual(prepared.hits,1);self.assertEqual(prepared.prepared_fallbacks,1)
            for row in range(8):prepared.prefetch(0,torch.tensor([row]))
            self.assertLessEqual(len(prepared.pending),2)
            actual=prepared.fetch_dequant(0,torch.tensor([15]),torch.float16)
            self.assertTrue(torch.equal(actual,eg.dequant_rows(store.raw[[15]],torch.float16)))
        finally:prepared.close()
        self.assertFalse(prepared.pending)
        with self.assertRaises(ValueError):eg.PrefetchRowStore(store,max_pending=0)

    def test_disabled_raw_fetch_contract(self):
        store=Store();prepared=eg.PrefetchRowStore(store,prepared_cpu=False)
        try:
            ids=torch.tensor([3,4,3]);prepared.prefetch(0,ids)
            self.assertTrue(torch.equal(prepared.fetch(0,ids),store.raw[ids]))
            self.assertEqual(prepared.prepared_hits,0)
        finally:prepared.close()


if __name__=='__main__':unittest.main()
