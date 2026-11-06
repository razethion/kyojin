"""Exercise actual PrefillEngine env parsing and busy-window cap without Torch."""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SRC=Path(__file__).resolve().parents[1]/"tools/multinode/prefill_core.py"

class Ids:
    def __init__(self,n):self.shape=(1,n)
    def __getitem__(self,key):return Ids(key[1].stop-key[1].start)

class Base:
    def __init__(self,**kwargs):
        self.pipeline=SimpleNamespace(failed=False);self.prefill_chunk=256
        self.active={"s":1};self.prefix_pool=None;self.lock=nullcontext();self.closed=False
    def _event(self,r,**kwargs):return kwargs
    def _fail(self):raise AssertionError("unexpected fail")
    def _finish(self,*a):raise AssertionError("unexpected finish")
    def step(self,*a):raise AssertionError("unexpected decode fallback")

def build(recorder):
    tree=ast.parse(SRC.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="PrefillEngine")
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ("__init__","step")]
    ns=dict(os=os,GreedyEngine=Base,prefill_pipeline=recorder)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(SRC),"exec"),ns)
    return ns["PrefillEngine"]

class EngineFlags(unittest.TestCase):
    def run_case(self,active):
        calls=[]
        def pipeline(pipe,name,chunks,**kwargs):calls.append((chunks,kwargs));return None,len(chunks)
        cls=build(pipeline)
        with patch.dict(os.environ,{"DS41_CLUSTER_PREFILL_WINDOW":"16","DS41_CLUSTER_PREFILL_NATIVE":"1"},clear=True):
            engine=cls()
        engine.active={str(i):1 for i in range(active)}
        request=SimpleNamespace(done=False,offset=0,ids=Ids(5001),cancelled=SimpleNamespace(is_set=lambda:False),name="s")
        result=engine.step(request)
        return calls,request,result

    def test_native_bundle_single_session_window16(self):
        calls,r,result=self.run_case(1)
        self.assertEqual(len(calls[0][0]),16)
        self.assertEqual(r.offset,4096)
        self.assertEqual(result,{"phase":"prefill"})
        options=calls[0][1]
        self.assertTrue(options["state_only"]);self.assertTrue(options["prefetch_ahead"])
        self.assertEqual(options["hid_rows"],128)

    def test_two_active_sessions_cap_window4(self):
        calls,r,_=self.run_case(2)
        self.assertEqual(len(calls[0][0]),4)
        self.assertEqual(r.offset,1024)
        self.assertTrue(all(piece.shape[1]==256 for piece in calls[0][0]))

if __name__=="__main__":unittest.main()
