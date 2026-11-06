"""CPU scheduling/order/error gates; no torch/native dependency."""
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tools.multinode.prefill_pipeline import draft_verify_jobs

class Ids:
    shape=(1,1)

class Sessions:
    def __init__(self):
        self.items={n:SimpleNamespace(step=3,length=9) for n in ('x','y')}
        self.begun=[];self.ready=[]
    def begin(self,*args):self.begun.append(args)
    def mark_ready(self,n):self.ready.append(n)

class PairSchedule(unittest.TestCase):
    def pipe(self,fail=False):
        draft2=threading.Event();a1=threading.Event();b1=threading.Event()
        log=[]
        def draft(name,*args,**kwargs):
            log.append('draft'+name)
            if name=='y':
                draft2.set();self.assertTrue(a1.wait(2))
            return {'tokens':Ids()}
        def local(op,args):
            name=args['session'];log.append('A'+name)
            if name=='x':
                self.assertTrue(draft2.wait(2));a1.set()
            else:
                self.assertTrue(b1.wait(2))
                if fail:raise RuntimeError('injected local failure')
            return name
        def remote(op,args):
            name=args['session'];log.append('B'+name)
            if name=='x':b1.set()
            return {'tokens':name}
        p=SimpleNamespace(sessions=Sessions(),depth=3,draft=draft,
                          clients=[SimpleNamespace(call=local),SimpleNamespace(call=remote)],closed=False)
        p.close=lambda:setattr(p,'closed',True)
        return p,log
    def run_pair(self,p):
        def cat(xs,dim):
            out=Ids();out.shape=(1,2);return out
        with patch.dict(sys.modules,{'torch':SimpleNamespace(cat=cat)}):
            return draft_verify_jobs(p,[SimpleNamespace(name=n,current=Ids()) for n in ('x','y')])
    def test_actual_overlap_and_whole_block_journals(self):
        p,log=self.pipe();jobs,proposals,results=self.run_pair(p)
        self.assertEqual(p.sessions.begun,[('x',3,9,2),('y',3,9,2)])
        self.assertEqual(p.sessions.ready,['x','y'])
        self.assertEqual(set(results),{'x','y'});self.assertFalse(p.closed)
        self.assertLess(log.index('drafty'),log.index('Bx'))
        self.assertLess(log.index('Bx'),log.index('By'))
    def test_error_drains_and_closes(self):
        p,_=self.pipe(True)
        with self.assertRaisesRegex(RuntimeError,'injected'):self.run_pair(p)
        self.assertTrue(p.closed);self.assertEqual(p.sessions.ready,[])
    def test_depth_refused_before_mutation(self):
        p,_=self.pipe();p.depth=8
        with self.assertRaises(ValueError):draft_verify_jobs(p,[SimpleNamespace(name=n) for n in ('x','y')])
        self.assertEqual(p.sessions.begun,[])
    def test_duplicate_refused_before_mutation(self):
        p,_=self.pipe()
        with self.assertRaises(ValueError):draft_verify_jobs(p,[SimpleNamespace(name='x')]*2)
        self.assertEqual(p.sessions.begun,[])

if __name__=='__main__':unittest.main()
