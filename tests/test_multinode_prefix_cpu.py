"""Native rings/compressors/hash/draft state across requests and changed prompts."""
import unittest
import torch
from test_multinode_ds41_cpu import world, Local, DeepSeekStage, SpeculativePipeline, MODE
from tools.multinode.generation import GreedyEngine
from tools.multinode.ds41_prefix import PrefixPool
from tools.multinode.round_policy import RoundPolicy


class PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg,cls.full,cls.a,cls.b=world()

    def core(self, entries=8,budget=8<<20,token_budget=None):
        stages=[DeepSeekStage(m,model_id='tiny',source_id='same',context=256,
                  sessions=2,depth=4,max_rows=32) for m in (self.a,self.b)]
        pipe=SpeculativePipeline([Local(s) for s in stages],mode=MODE)
        pool=PrefixPool(pipe,entries=entries,budget=budget,grid=32,assistant_id=99,token_budget=token_budget)
        return GreedyEngine(pipe,vocab_size=512,prefill_chunk=16,adaptive_drafting=False,
                           prefix_pool=pool),stages

    def run_request(self,core,ids,n=16,cache=True):
        r=core.start(torch.tensor([ids]),max_new_tokens=n,cache_prompt=cache)
        out=[]
        while not r.done: out+=r.step()['token_ids']
        return out,r

    def test_native_repeat_extension_changed_thinking_and_compaction(self):
        core,stages=self.core()
        ids=[3,5,7,10]*27+[99,12,13]
        cold,_=self.run_request(core,ids,cache=False)
        warm,_=self.run_request(core,ids)
        repeat,r=self.run_request(core,ids)
        self.assertEqual(cold,warm);self.assertEqual(cold,repeat)
        self.assertEqual(r.cached_tokens,len(ids)-1)
        extended=ids+warm+[28,99,30]
        expected,_=self.run_request(core,extended,cache=False)
        actual,r=self.run_request(core,extended)
        self.assertEqual(actual,expected);self.assertGreaterEqual(r.cached_tokens,len(ids)-3)
        # A render drops earlier thinking: resume before the changed suffix.
        changed=ids[:-2]+[44,45,46]
        expected,_=self.run_request(core,changed,cache=False)
        actual,r=self.run_request(core,changed)
        self.assertEqual(actual,expected);self.assertGreater(r.cached_tokens,0)
        compacted=[88]+ids[30:]
        expected,_=self.run_request(core,compacted,cache=False)
        actual,r=self.run_request(core,compacted)
        self.assertEqual(actual,expected);self.assertEqual(r.cached_tokens,0)
        core.close()

    def test_two_slots_cancel_reset_and_bounded_eviction(self):
        core,stages=self.core(entries=3)
        a=core.start(torch.tensor([[3]*81]),max_new_tokens=20)
        b=core.start(torch.tensor([[5]*85]),max_new_tokens=20)
        for _ in range(8):
            if not a.done:a.step()
            if not b.done:b.step()
        a.cancel();a.close();b.close()
        self.assertLessEqual(len(core.prefix_pool.items),3)
        self.assertTrue(all(len(s.prefixes)<=3 for s in stages))
        core.clear_prefix_cache()
        self.assertFalse(core.prefix_pool.items)
        self.assertTrue(all(not s.prefixes for s in stages))
        _,r=self.run_request(core,[3]*81)
        self.assertEqual(r.cached_tokens,0)
        core.close()

    def test_budget_eviction_and_adaptive_tokens(self):
        core,stages=self.core(budget=1)
        ids=[3,8,2,4]*20
        core.pipeline.depth=0
        reference,_=self.run_request(core,ids,32,cache=False)
        core.pipeline.depth=3
        core.policy_factory=lambda:RoundPolicy(3)
        adaptive,r=self.run_request(core,ids,32)
        self.assertEqual(reference,adaptive)
        self.assertFalse(core.prefix_pool.items)
        self.assertTrue(all(not s.prefixes for s in stages))
        self.assertGreater(r.profile['draft_steps'],0)
        self.assertGreater(r.profile['rounds'],r.profile['draft_steps'])
        core.close()

    def test_prefix_over_authenticated_tcp(self):
        from contextlib import ExitStack
        from test_multinode_cpu import worker
        from tools.multinode.worker import StageClient
        stages=[DeepSeekStage(m,model_id='tiny',source_id='same',context=256,
                  sessions=2,depth=4,max_rows=32) for m in (self.a,self.b)]
        with ExitStack() as stack:
            clients=[]
            for stage in stages:
                port,token,_=stack.enter_context(worker(stage))
                clients.append(StageClient('127.0.0.1',port,token))
            pipe=SpeculativePipeline(clients,mode=MODE)
            core=GreedyEngine(pipe,vocab_size=512,prefill_chunk=16,adaptive_drafting=False,
                             prefix_pool=PrefixPool(pipe,grid=32))
            expected,_=self.run_request(core,[5,8,12]*24)
            actual,r=self.run_request(core,[5,8,12]*24)
            self.assertEqual(expected,actual);self.assertEqual(r.cached_tokens,71)
            core.close()

    def test_retained_token_budget_preserves_repeat_and_bounds_distinct_prompts(self):
        core,stages=self.core(token_budget=120)
        first=[3,5,7]*27
        expected,_=self.run_request(core,first,cache=False)
        self.run_request(core,first)
        actual,repeat=self.run_request(core,first)
        self.assertEqual(actual,expected)
        self.assertEqual(repeat.cached_tokens,len(first)-1)
        for token in (11,13,17):
            self.run_request(core,[token]*85)
            stats=core.prefix_pool.stats()
            self.assertLessEqual(stats['retained_tokens'],120)
            self.assertEqual(stats['retained_tokens'],sum(e['n'] for e in core.prefix_pool.items))
            self.assertEqual(stats['token_budget'],120)
        self.assertFalse(any(e['keys']==first[:-1] for e in core.prefix_pool.items))
        core.close()

    def test_zero_retained_token_budget_disables_saving(self):
        core,_=self.core(token_budget=0)
        self.run_request(core,[3]*81)
        self.assertEqual(core.prefix_pool.stats()['retained_tokens'],0)
        self.assertFalse(core.prefix_pool.items)
        core.close()


class PolicyTests(unittest.TestCase):
    def test_cost_and_periodic_recovery_probe(self):
        p=RoundPolicy(3)
        choices=[]
        for _ in range(64):
            d=p.choose();choices.append(d)
            p.observe(d,.1 if d==0 else .5,1 if d==0 else 2)
        self.assertGreater(choices[6:].count(0),25)
        self.assertTrue(any(d==3 for d in choices[16:]))
