"""Accepted-prefix transactions, replay rejection and fail-stop without a GPU."""
import unittest
from tools.multinode.session import Sessions
from tools.multinode.speculative import SpeculativePipeline,MODE,GLM_MODE,greedy_prefix,SpeculativeGlmStage

class Client:
    def __init__(self,begin,end):
        self.manifest=dict(protocol=1,mode=MODE,model_id='weights',source_id='source',total_modules=4,
                           begin=begin,end=end,context=512,sessions=2,cache_bits=8,max_history=7)
        self.closed=False;self.state=Sessions(512,2);self.fail=False
    def call(self,op,args):
        name=args['session']
        if op=='open':return self.state.open(name)
        if op=='prepare':
            self.state.begin(name,args['step'],args['past'],args['input_ids'].shape[1]);self.state.mark_ready(name)
            return None
        if op=='commit':
            if self.fail:raise OSError('injected failed commit')
            return self.state.commit(name,args['step'],args['accepted'])
        if op=='release':return self.state.release(name)
        raise ValueError(op)
    def close(self):self.closed=True
    def execute(self,op,args):return self.call(op,args)

class IDs:
    shape=(1,8)

class SpecTests(unittest.TestCase):
    def test_glm_fixed_depth_does_not_require_qwen_probability_head(self):
        import torch
        stage=object.__new__(SpeculativeGlmStage)
        stage.final=True;stage.depth=2;stage.vocab=100
        stage.sessions=Sessions(512,1);stage.sessions.open('a')
        stage.sessions.items['a'].length=4
        stage.carry={'a':torch.zeros(1,1,8)};stage.draft_pending={}
        class Draft:
            def forward(self,ids,params):return params['hidden']+1
            def sample_from_state(self,hidden,params):return torch.tensor([[7]])
        stage.draft_model=Draft()
        stage.draft_params=lambda name,past,hidden:dict(hidden=hidden)
        values=dict(session='a',step=0,past=4,input_ids=torch.tensor([[5]]),depth=2,adaptive=True)
        with self.assertRaisesRegex(ValueError,'fixed-depth'):stage.draft(values)
        values['adaptive']=False
        actual=stage.draft(values)
        self.assertEqual(actual['tokens'].tolist(),[[7,7]])
        self.assertEqual(actual['probabilities'],[None,None])
        self.assertEqual(stage.draft_pending['a'][:2],(0,4))

    def test_glm_mode_uses_distinct_architecture_handshake(self):
        from tools.multinode.stage import CachedGlmStage
        self.assertTrue(issubclass(SpeculativeGlmStage,CachedGlmStage))
        self.assertEqual(SpeculativeGlmStage.architectures,CachedGlmStage.architectures)
        self.assertFalse(SpeculativeGlmStage.uses_qwen_tier)
        clients=[Client(0,2),Client(2,4)]
        for client in clients:client.manifest['mode']=GLM_MODE
        pipe=SpeculativePipeline(clients,mode=GLM_MODE)
        pipe.open('a');pipe.prepare('a',IDs(),speculative=True)
        self.assertEqual(pipe.commit('a',3),{'length':3,'step':1})
        pipe.release('a');pipe.close()
        with self.assertRaises(ValueError):SpeculativePipeline(clients)

    def test_greedy_commit_preserves_unprocessed_final_token(self):
        cases=[([], [9], 1, [9], 1, 0),
               ([5,6], [9,6,7], 8, [9], 1, 0),
               ([5,6], [5,9,7], 8, [5,9], 2, 1),
               ([5,6], [5,6,7], 8, [5,6,7], 3, 2),
               ([5,6], [5,6,7], 1, [5], 1, 2),
               ([5,6], [5,6,7], 2, [5,6], 2, 2)]
        for drafts,predictions,budget,emitted,committed,accepted in cases:
            self.assertEqual(greedy_prefix(drafts,predictions,budget),
                             dict(emitted=emitted,committed=committed,accepted=accepted))
        for values in (([5],[5],2),([5],[5,6],True),([5],[5,-1],2),([True],[5,6],2)):
            with self.assertRaises(ValueError):greedy_prefix(*values)

    def test_prefix_commit_over_real_authenticated_tcp_workers(self):
        import torch
        from test_multinode_cpu import worker
        from tools.multinode.worker import StageClient
        first,last=Client(0,2),Client(2,4)
        with worker(first) as (p0,t0,_),worker(last) as (p1,t1,_):
            pipe=SpeculativePipeline([StageClient('127.0.0.1',p0,t0),StageClient('127.0.0.1',p1,t1)])
            try:
                pipe.open('a');pipe.prepare('a',torch.zeros((1,8),dtype=torch.int64),speculative=True)
                self.assertEqual(pipe.commit('a',3),{'length':3,'step':1})
                self.assertEqual([first.state.items['a'].length,last.state.items['a'].length],[3,3])
                pipe.release('a')
            finally:pipe.close()

    def test_every_accepted_prefix_and_no_replay(self):
        for accepted in range(1,9):
            ledger=Sessions(512,1);ledger.open('a');ledger.begin('a',0,0,8);ledger.mark_ready('a')
            self.assertEqual(ledger.commit('a',0,accepted),{'length':accepted,'step':1})
            with self.assertRaises(ValueError):ledger.commit('a',0,accepted)
            ledger.begin('a',1,accepted,1)

    def test_invalid_prefix_does_not_change_pending_or_position(self):
        ledger=Sessions(512,1);ledger.open('a');ledger.begin('a',0,0,8);ledger.mark_ready('a')
        for accepted in (0,-1,9,True,1.5):
            with self.assertRaises(ValueError):ledger.commit('a',0,accepted)
            self.assertEqual(ledger.items['a'].length,0);self.assertTrue(ledger.items['a'].ready)
        self.assertEqual(ledger.commit('a',0,3)['length'],3)

    def test_two_nodes_agree_on_prefix_and_next_step(self):
        clients=[Client(0,2),Client(2,4)];pipe=SpeculativePipeline(clients);pipe.open('a')
        pipe.prepare('a',IDs(),speculative=True)
        self.assertEqual(pipe.commit('a',3),{'length':3,'step':1})
        self.assertEqual([c.state.items['a'].length for c in clients],[3,3])
        pipe.prepare('a',IDs(),speculative=True);pipe.commit('a',8);pipe.release('a')
        self.assertTrue(all(not c.state.items for c in clients))

    def test_partial_commit_failure_poison_all_without_retry(self):
        clients=[Client(0,2),Client(2,4)];pipe=SpeculativePipeline(clients);pipe.open('a')
        pipe.prepare('a',IDs(),speculative=True);clients[-1].fail=True
        with self.assertRaises(OSError):pipe.commit('a',3)
        self.assertTrue(all(c.closed for c in clients));self.assertTrue(pipe.failed)
        with self.assertRaises(RuntimeError):pipe.commit('a',3)
        self.assertEqual(pipe.sessions.items['a'].length,0)
        self.assertEqual(clients[0].state.items['a'].length,3)
        self.assertEqual(clients[-1].state.items['a'].length,0)

    def test_pending_prepare_cannot_release_or_draft(self):
        for op in ('release','draft'):
            clients=[Client(0,2),Client(2,4)];pipe=SpeculativePipeline(clients);pipe.open('a')
            pipe.prepare('a',IDs(),speculative=True)
            with self.assertRaises(ValueError):
                pipe.release('a') if op=='release' else pipe.draft('a',None,2)
            self.assertTrue(all(c.closed for c in clients))

if __name__=='__main__':unittest.main(verbosity=2)
