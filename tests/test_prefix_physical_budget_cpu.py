"""Coordinator budgeting uses unique retained storage, including eviction lineage changes."""
import gc
from types import SimpleNamespace
import unittest
import weakref
import torch
from exllamav3.dsv41 import prefix_cache as pc
from tools.multinode.ds41_prefix import PrefixPool


class Client:
    def __init__(self,pages):
        self.pages=pages;self.entries={};self.previous=None;self.n=0
        self.cfg=SimpleNamespace(ratio=lambda layer:1)
        self.st=SimpleNamespace(win={0:torch.zeros(1,1,4)},
            comp_kv={0:torch.arange(128*16.).reshape(1,128,16)},idx_k={0:torch.zeros(1,128,4)},
            comp_kv_state={},comp_score_state={},hasher=SimpleNamespace(cache=torch.arange(128).reshape(1,128)))
    def call(self,op,args):
        snaps=list(self.entries.values());physical=pc.unique_snapshot_bytes(snaps)
        if op=='prefix_stats':return dict(physical_bytes=physical)
        if op=='prefix_estimate':return dict(physical_bytes=physical,increment_bytes=pc.snapshot_increment_bytes(
            self.cfg,self.st,self.n,previous=self.entries.get(self.previous),page_rows=self.pages))
        if op=='prefix_save':
            snap=pc.snapshot(self.cfg,self.st,self.n,(),None,'cluster',page_rows=self.pages,
                previous=self.entries.get(self.previous),trusted_previous=True)
            self.entries[args['key']]=snap;self.previous=args['key']
            return dict(saved=True,bytes=snap.nbytes)
        if op=='prefix_drop':
            self.entries.pop(args['key'],None)
            if self.previous==args['key']:self.previous=None
            return {}
        if op=='prefix_clear':self.entries.clear();self.previous=None;return {}
        raise ValueError(op)


class PhysicalBudgetTests(unittest.TestCase):
    def test_sharing_retains_more_checkpoints_and_eviction_releases_pages(self):
        retained=[]
        for pages in (0,16):
            clients=[Client(pages),Client(pages)]
            pool=PrefixPool(SimpleNamespace(clients=clients,close=lambda:None),budget=14000,entries=8)
            request=SimpleNamespace(cache_prompt=True,name='same',fed=[])
            for n in (32,64,96,128):
                request.fed=list(range(n))
                for client in clients:client.n=n
                pool.save(request)
                self.assertTrue(all(size<=14000 for size in pool.stats()['bytes_per_node']))
            retained.append(len(pool.items))
            self.assertEqual(pool.stats()['bytes_per_node'],
                [pc.unique_snapshot_bytes(c.entries.values()) for c in clients])
            if pages:
                self.assertEqual(len(pool.items),4)
                self.assertGreater(pool.stats()['logical_bytes_per_node'][0],14000)
                page=next(iter(clients[0].entries.values())).tensors['comp_kv'][0].pages[0]
                ref=weakref.ref(page);del page
                # Re-estimation after dropping lineage must reject an oversized
                # standalone snapshot instead of trusting the old small increment.
                pool.budget=1000;request.fed=list(range(127))
                for client in clients:client.n=127
                pool.save(request);gc.collect()
                self.assertFalse(pool.items);self.assertIsNone(ref())
                self.assertEqual(pool.stats()['bytes_per_node'],[0,0])
        self.assertGreater(retained[1],retained[0])


if __name__=='__main__':unittest.main()
