"""Weakref proof: release/transport teardown must not depend on cyclic GC."""
import gc
from unittest.mock import patch
import unittest
import weakref
import torch
from test_multinode_ds41_cpu import world,DeepSeekStage
from test_multinode_cpu import worker
from tools.multinode.worker import StageClient


class SessionLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg,cls.model,cls.a,cls.b=world()

    def stage(self):
        return DeepSeekStage(self.model,model_id='tiny',source_id='same',context=256,sessions=2,depth=4,max_rows=32)

    def populate(self,stage):
        stage.execute('open',{'session':'live'})
        stage.states['live'].ensure(32);stage.states['live'].length=32
        stage.sessions.items['live'].length=32
        stage.carry['live']=torch.ones(1,4,self.cfg.hidden_size)
        stage.specs['live'].rows=stage.carry['live']
        with patch.dict('os.environ',{'DS41_PFX_PAGE_ROWS':'4'}):
            self.assertTrue(stage.execute('prefix_save',{'session':'live','key':'a'*32})['saved'])
        cache=stage.states['live'];spec=stage.specs['live']
        refs=[weakref.ref(item) for item in (cache,cache.hasher,cache.hasher.cache,
              next(iter(cache.win.values())),stage.carry['live'],spec,spec.target,spec.drafter)]
        page=next(iter(stage.prefixes['a'*32][0].tensors['comp_kv'].values())).pages[0]
        return refs,weakref.ref(page)

    def test_normal_release_frees_cache_cycle_without_gc_keeps_host_prefix(self):
        stage=self.stage();enabled=gc.isenabled();gc.disable()
        try:
            refs,page=self.populate(stage)
            stage.execute('release',{'session':'live'})
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertIsNotNone(page())
            self.assertGreater(stage.execute('prefix_stats',{})['physical_bytes'],0)
            stage.execute('prefix_clear',{})
            self.assertIsNone(page())
            stage.close();stage.close()
        finally:
            stage.close()
            if enabled:gc.enable()

    def test_close_abandons_native_journal_and_all_session_tensors_without_gc(self):
        stage=self.stage();enabled=gc.isenabled();gc.disable()
        try:
            refs,page=self.populate(stage)
            stage.states['live'].begin_block(32,3)
            stage.sessions.items['live'].pending=(0,32,3)
            stage.pending['live']=(3,True,torch.ones(1,3,self.cfg.hidden_size))
            stage.draft_pending['live']=(0,32,torch.tensor([[7]]))
            stage.close();stage.close()
            self.assertTrue(all(ref() is None for ref in refs));self.assertIsNone(page())
            self.assertFalse(stage.sessions.items or stage.states or stage.specs or stage.pending or stage.draft_pending)
            with self.assertRaises(RuntimeError):stage.execute('open',{'session':'new'})
        finally:
            stage.close()
            if enabled:gc.enable()

    def test_authenticated_disconnect_and_execute_exception_close_ownership(self):
        for failure in ('disconnect','execute_error','native_error'):
            with self.subTest(failure=failure):
                stage=self.stage();enabled=gc.isenabled();gc.disable()
                try:
                    refs,page=self.populate(stage)
                    with worker(stage) as (port,token,errors):
                        client=StageClient('127.0.0.1',port,token)
                        if failure=='execute_error':
                            with self.assertRaises(RuntimeError):client.call('prefix_restore',{'session':'live','key':'a'*32})
                        if failure=='native_error':
                            def fail(*args,**kwargs):raise RuntimeError('injected after native journal allocation')
                            with patch.object(stage.model,'forward_stage',new=fail):
                                with self.assertRaises(RuntimeError):
                                    client.call('prepare',dict(session='live',step=0,past=32,input_ids=torch.tensor([[7,8,9]]),
                                        activation=None,last_tokens_only=3,speculative=True,return_logits=True))
                        client.close()
                    self.assertTrue(stage.poisoned)
                    self.assertTrue(all(ref() is None for ref in refs));self.assertIsNone(page())
                finally:
                    stage.close()
                    if enabled:gc.enable()


if __name__=='__main__':unittest.main()
