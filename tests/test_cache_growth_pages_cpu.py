"""Storage optimizations: preservation, lineage, partial groups, physical bytes."""
import unittest
from types import SimpleNamespace
import torch
from exllamav3.dsv41.serve_cache import PagedPool
from exllamav3.dsv41 import prefix_cache as pc


class CacheStorageTests(unittest.TestCase):
    def test_bounded_growth_preserves_every_visible_row_and_fill(self):
        for growth in ('linear', 'geometric'):
            p = PagedPool(2, 3, torch.float32, 'cpu', 19, page=4, fill=-7, growth=growth)
            for n in range(1, 20):
                before = p.t.clone()
                p.ensure(n)
                self.assertLessEqual(p.capacity, 19)
                self.assertTrue(torch.equal(p.t[:, :before.size(1)], before))
                self.assertTrue((p.t[:, before.size(1):] == -7).all())
                p.t[:, n-1] = n
            self.assertEqual(p.capacity, 19)
            self.assertEqual(p.nbytes(), 2*19*3*4)

    def state(self):
        return SimpleNamespace(win={0:torch.ones(1,4,2)},
            comp_kv={0:torch.arange(24.).reshape(1,12,2)},
            idx_k={0:torch.arange(12.).reshape(1,12,1)},
            comp_kv_state={0:torch.ones(1,2,2)},
            comp_score_state={0:torch.ones(1,2,2)},
            hasher=SimpleNamespace(cache=torch.arange(24).reshape(1,24)))

    def test_geometric_reduces_relocations(self):
        relocations={}
        for growth in ('linear','geometric'):
            p=PagedPool(1,1,torch.float32,'cpu',128,page=4,growth=growth)
            count=0
            for rows in range(1,129):
                old=p.t
                p.ensure(rows)
                count+=old is not p.t
            relocations[growth]=count
        self.assertEqual(relocations,{'linear':32,'geometric':6})

    def test_completed_page_sharing_partial_group_and_restore(self):
        cfg=SimpleNamespace(ratio=lambda l:2)
        st=self.state()
        a=pc.snapshot(cfg,st,9,list(range(9)),torch.ones(2,2),'grid',page_rows=2)
        self.assertEqual(pc.snapshot_bytes(cfg,st,9,torch.ones(2,2)),a.nbytes)
        b=pc.snapshot(cfg,st,13,list(range(13)),torch.ones(2,2),'grid',previous=a,page_rows=2)
        ap=a.tensors['comp_kv'][0].pages;bp=b.tensors['comp_kv'][0].pages
        self.assertIs(ap[0],bp[0]);self.assertIs(ap[1],bp[1])
        self.assertIsNot(ap[2],bp[2])  # row4 belongs to the unfinished ratio2 group at n9
        self.assertLess(pc.unique_snapshot_bytes([a,b]),a.nbytes+b.nbytes)
        self.assertEqual(pc.unique_snapshot_bytes([a]),a.nbytes)
        st.win[0].zero_();st.comp_kv[0].zero_();st.comp_kv_state[0].zero_();st.hasher.cache.zero_()
        self.assertTrue((a.tensors['win'][0]==1).all())
        self.assertTrue((a.tensors['comp_kv_state'][0]==1).all())
        self.assertEqual(a.tensors['hasher'][0,8].item(),8)
        dst=self.state();dst.ensure=lambda n:None;dst.length=0
        model=SimpleNamespace(new_state=lambda batch,seq:dst)
        pc.restore(model,b,24)
        self.assertTrue(torch.equal(dst.comp_kv[0][:,:7],torch.arange(14.).reshape(1,7,2)))
        self.assertEqual(dst.length,13)

    def test_empty_or_changed_keys_do_not_establish_lineage(self):
        cfg=SimpleNamespace(ratio=lambda l:2);st=self.state()
        for keys in ([],list(range(8))):
            a=pc.snapshot(cfg,st,8,keys,None,'grid',page_rows=2)
            b=pc.snapshot(cfg,st,12,[99]*12,None,'grid',previous=a,page_rows=2)
            self.assertIsNot(a.tensors['comp_kv'][0].pages[0],b.tensors['comp_kv'][0].pages[0])

    def test_changed_page_geometry_never_reuses_mispositioned_tail(self):
        cfg=SimpleNamespace(ratio=lambda l:1);st=self.state()
        a=pc.snapshot(cfg,st,5,list(range(5)),None,'grid',page_rows=2)
        b=pc.snapshot(cfg,st,7,list(range(7)),None,'grid',page_rows=1,previous=a)
        self.assertEqual(pc.snapshot_increment_bytes(cfg,st,7,previous=a,page_rows=1),b.nbytes)
        self.assertTrue(torch.equal(b.tensors['comp_kv'][0].pages[2],st.comp_kv[0][:,2:3]))


if __name__=='__main__':unittest.main()
