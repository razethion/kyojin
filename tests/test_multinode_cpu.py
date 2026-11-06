"""CPU-only transport/span proof; no extension, checkpoint or GPU required.

Run directly: PYTHONPATH=. python tests/test_multinode_cpu.py
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import tempfile
from types import SimpleNamespace
import unittest

import torch

from tools.multinode.wire import Channel, ProtocolError, pack, unpack, MAX_PAYLOAD, PREFIX, MAGIC
from tools.multinode.stage import DenseStage, CachedQwenStage
from tools.multinode.worker import Pipeline, CachedPipeline, StageClient, serve_connection
from tools.multinode.session import Sessions
from tools.multinode.identity import manifest, verify
from tools.multinode.link_probe import echo_connection, measure
from tools.multinode.network import validate_binding


class ToyModule:
    def __init__(self, index):
        self.index = index
        self.key = f"module.{index}"
        self.caps = {"logits_output": True} if index == 4 else {}
        self.loaded = False

    def prepare_for_device(self, x, params):
        assert self.loaded
        return x

    def forward(self, x, params):
        if self.index == 0:
            return x.unsqueeze(-1).float() + torch.arange(8).float()
        if self.index == 4:
            return torch.cat([x, x.sum(-1, keepdim=True)], dim=-1)
        return x * (self.index + 1) + self.index


class ToyModel:
    def __init__(self):
        self.config = SimpleNamespace(architecture="LlamaForCausalLM", layer_map=None, hidden_size=8)
        self.loaded_tp = False
        self.recurrent_state_cls = None
        self.cache_weakrefs = {}
        self.modules = [ToyModule(i) for i in range(5)]
        self.fwd_modules = [(m, 0, i) for i, m in enumerate(self.modules)]
        self.loaded_keys = []

    def load(self, **kwargs):
        for module in self.modules:
            module.loaded = True
            self.loaded_keys.append(module.key)

    def prepare_inputs(self, ids, params):
        return ids


def toy_stage(begin, end):
    stage = DenseStage(ToyModel(), begin, end, device="cpu", model_id="seeded-toy-weights-v1",
                       source_id="synthetic-source-v1", max_rows=64)
    stage.load()
    return stage


class CachedToy:
    def __init__(self,begin,end,fail=None):
        self.stage=toy_stage(begin,end)
        self.manifest=dict(self.stage.manifest,mode='cached-dense-v1',context=512,sessions=2)
        self.sessions=Sessions(512,2)
        self.closed=False
        self.fail=fail
        self.ops=[]

    def call(self,op,args):
        self.ops.append(op)
        if self.fail==op:raise OSError('injected stage failure')
        if op=='open':return self.sessions.open(args['session'])
        if op=='commit':return self.sessions.commit(args['session'],args['step'])
        if op=='release':return self.sessions.release(args['session'])
        if op=='prepare':
            self.sessions.begin(args['session'],args['step'],args['past'],args['input_ids'].shape[1])
            out=self.stage.forward({key:args[key] for key in ('input_ids','activation','last_tokens_only')})
            self.sessions.mark_ready(args['session'])
            return out
        raise ValueError('unknown operation')

    def execute(self,op,args):return self.call(op,args)
    def close(self):self.closed=True


@contextlib.contextmanager
def worker(stage, token=None):
    token = token or os.urandom(32)
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0)); listener.listen(1)
        port = listener.getsockname()[1]

        def run():
            try:
                sock, _ = listener.accept()
                serve_connection(sock, token, stage)
            except (EOFError, OSError, ProtocolError) as e:
                errors.append(e)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield port, token, errors
        finally:
            thread.join(timeout=2)
            if thread.is_alive():
                raise AssertionError("worker thread leaked")


class WireTests(unittest.TestCase):
    def test_qwen_deferred_update_materialized_at_stage_boundary(self):
        stage=object.__new__(CachedQwenStage)
        stage.model=ToyModel();stage.model.load()
        stage.model.modules=stage.model.modules[:2]
        stage.model.fwd_modules=stage.model.fwd_modules[:2]
        stage.manifest={'begin':0};stage.max_rows=256
        module=stage.model.modules[-1];original=module.forward
        def deferred(x,params):
            result=original(x,params)
            params['hc_pending']=(result,torch.full_like(result,7),None,None)
            return result
        module.forward=deferred
        def flush(x,pending):
            self.assertIs(x,pending[0]);x.add_(pending[1])
        stage.flush_pending=flush
        ids=torch.tensor([[2,3]])
        actual=stage._forward({'input_ids':ids,'activation':None,'last_tokens_only':1},{})
        expected=(ids.unsqueeze(-1).float()+torch.arange(8).float())*2+1+7
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(stage.boundary_flushes,1)

    def test_qwen_slot_clear_uses_inference_context(self):
        with torch.inference_mode():buffer=torch.ones(4)
        class Cache:
            def get_new_state(self):
                buffer.zero_()
                return SimpleNamespace(position=0)
        stage=object.__new__(CachedQwenStage)
        stage.poisoned=False;stage.sessions=Sessions(256,1);stage.cache=Cache()
        stage.states={};stage.pending_recurrent={}
        self.assertEqual(stage.execute('open',{'session':'test'}),{'slot':0,'length':0,'step':0})
        self.assertTrue(torch.equal(buffer,torch.zeros(4)))

    def test_qwen_stream_boundary_keeps_fp32_rank_and_geometry(self):
        stage=object.__new__(CachedQwenStage)
        stage.stream_input=True
        stage.model=SimpleNamespace(config=SimpleNamespace(hc_mult=4,hidden_size=8))
        ids=torch.ones((1,3),dtype=torch.long)
        self.assertTrue(stage.valid_activation(torch.ones((1,3,4,8),dtype=torch.float32),ids))
        for value in (torch.ones((1,3,4,8),dtype=torch.float16),torch.ones((1,3,8)),torch.ones((1,3,2,8)),torch.ones((1,4,4,8))):
            self.assertFalse(stage.valid_activation(value,ids))

    def test_tcp_frames_disable_nagle(self):
        with worker(toy_stage(0,5)) as (port,token,_):
            client=StageClient('127.0.0.1',port,token)
            self.assertEqual(client.channel.sock.getsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY),1)
            client.forward(torch.ones((1,1),dtype=torch.long))
            client.close()

    def test_cached_sessions_across_two_tcp_workers(self):
        a,b=CachedToy(0,2),CachedToy(2,5)
        with worker(a) as (port_a,token_a,_),worker(b) as (port_b,token_b,_):
            pipeline=CachedPipeline([StageClient('127.0.0.1',port_a,token_a),StageClient('127.0.0.1',port_b,token_b)])
            try:
                pipeline.open('a');pipeline.open('b')
                for name,ids in [('a',torch.arange(7).view(1,7)),('b',torch.arange(16).view(1,16)),('a',torch.tensor([[9]]))]:
                    expected=toy_stage(0,5).forward(dict(input_ids=ids,activation=None,last_tokens_only=1))
                    self.assertTrue(torch.equal(pipeline.forward(name,ids),expected))
                self.assertEqual(pipeline.sessions.items['a'].length,8)
                self.assertEqual(pipeline.sessions.items['b'].length,16)
                pipeline.release('a');pipeline.open('c')
            finally:pipeline.close()

    def test_cached_partial_prepare_or_commit_poison_all_stages(self):
        for phase in ('prepare','commit'):
            a,b=CachedToy(0,2),CachedToy(2,5,fail=phase)
            pipeline=CachedPipeline([a,b]);pipeline.open('a')
            with self.assertRaises(OSError):pipeline.forward('a',torch.tensor([[7]]))
            self.assertTrue(a.closed and b.closed and pipeline.failed)
            with self.assertRaises(RuntimeError):pipeline.forward('a',torch.tensor([[7]]))
            if phase=='prepare':self.assertNotIn('commit',a.ops)

    def test_cached_reject_mismatched_geometry(self):
        a,b=CachedToy(0,2),CachedToy(2,5)
        b.manifest['context']=256
        with self.assertRaises(ValueError):CachedPipeline([a,b])

    def test_worker_binding_requires_owned_usb4_address(self):
        inventory=[{'ifname':'thunderbolt0','addr_info':[{'local':'10.203.0.2'},{'local':'fe80::2'}]}]
        path='/sys/devices/pci/domain1/1-0/1-2/1-2.0'
        validate_binding('127.0.0.1')
        validate_binding('10.203.0.2','thunderbolt0',inventory=inventory,device_path=path)
        validate_binding('fe80::2%thunderbolt0','thunderbolt0',inventory=inventory,device_path=path)
        for host,interface,device in [('0.0.0.0',None,path),('10.203.0.2',None,path),
                                      ('10.203.0.3','thunderbolt0',path),('10.203.0.2','eno1',path),
                                      ('10.203.0.2','thunderbolt0','/sys/devices/pci/ethernet'),
                                      ('fe80::2%thunderbolt1','thunderbolt0',path)]:
            with self.assertRaises(ValueError):
                validate_binding(host,interface,inventory=inventory,device_path=device)

    def test_application_link_probe(self):
        token = os.urandom(32)
        failures = []
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); listener.listen(1)
            port = listener.getsockname()[1]
            def echo():
                try:
                    sock, _ = listener.accept(); echo_connection(sock, token)
                except EOFError: pass
                except Exception as e: failures.append(repr(e))
            thread = threading.Thread(target=echo, daemon=True); thread.start()
            rows = measure("127.0.0.1", port, token, [0, 1024, 1024**2], 2)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertFalse(failures)
            self.assertEqual([r['payload_bytes_each_direction'] for r in rows], [0, 1024, 1024**2])
            self.assertTrue(all(r['median_roundtrip_ms'] > 0 for r in rows))

    def test_tensor_bytes_all_dtypes_and_shapes(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64,
                      torch.int64, torch.int32, torch.uint8, torch.bool]:
            for shape in [(3, 4), (2, 0, 3), ()]:
                with self.subTest(dtype=dtype, shape=shape):
                    a = torch.arange(max(1, int(torch.tensor(shape).prod())))[:int(torch.tensor(shape).prod())].reshape(shape).to(dtype)
                    header, payload = pack({"x": a, "meta": (None, True, ["tensor", 2])})
                    b = unpack(header, payload)
                    self.assertEqual(b["x"].dtype, a.dtype)
                    self.assertEqual(b["x"].shape, a.shape)
                    self.assertEqual(b["x"].reshape(-1).view(torch.uint8).tolist(), a.reshape(-1).view(torch.uint8).tolist())
                    self.assertEqual(b["meta"], (None, True, ["tensor", 2]))

    def test_noncontiguous_and_multistream_shape(self):
        a = torch.arange(2 * 7 * 4 * 8).float().reshape(2, 7, 4, 8).transpose(1, 2)
        self.assertFalse(a.is_contiguous())
        b = unpack(*pack(a))
        self.assertTrue(torch.equal(a, b))

    def test_16_mib_tensor(self):
        a = torch.arange(4 * 1024**2).float()
        b = unpack(*pack(a))
        self.assertTrue(torch.equal(a, b))

    def test_unsupported_object_and_limit(self):
        with self.assertRaises(ProtocolError): pack(object())
        with self.assertRaises(ProtocolError): pack(torch.zeros(32), max_payload=16)
        with self.assertRaises(ProtocolError): pack(torch.zeros(1, dtype=torch.complex64))

    def test_bad_tensor_metadata(self):
        header, payload = pack(torch.ones(4))
        for change in [dict(size=2), dict(offset=1), dict(shape=[-1]), dict(dtype="pickle")]:
            meta = json.loads(header)
            meta["tensors"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ProtocolError):
                unpack(json.dumps(meta).encode(), payload)
        with self.assertRaises(ProtocolError): unpack(header, payload + b"junk")

    def test_wrong_authentication_key(self):
        with worker(toy_stage(0, 5)) as (port, _, errors):
            with self.assertRaises((EOFError, OSError, ProtocolError)):
                StageClient("127.0.0.1", port, os.urandom(32), timeout=2)
        self.assertTrue(errors)

    def test_partial_reads(self):
        class FragmentedSocket:
            def __init__(self, sock): self.sock = sock
            def recv_into(self, view): return self.sock.recv_into(view[:3])
            def sendall(self, data): self.sock.sendall(data)
            def close(self): self.sock.close()
        a, b = socket.socketpair()
        a.settimeout(2); b.settimeout(2)
        key, nonce = os.urandom(32), os.urandom(32)
        sender = Channel(a, key, nonce)
        receiver = Channel(FragmentedSocket(b), key, nonce, server=True)
        try:
            sender.send({"x": torch.arange(32)})
            self.assertTrue(torch.equal(receiver.recv()["x"], torch.arange(32)))
        finally: sender.close(); receiver.close()

    def test_truncated_frame(self):
        a, b = socket.socketpair()
        receiver = Channel(b, os.urandom(32), os.urandom(32))
        a.sendall(b"short"); a.close()
        try:
            with self.assertRaises(EOFError): receiver.recv()
        finally: receiver.close()

    def test_replay_and_oversized_frame_rejected_before_payload(self):
        for prefix in [PREFIX.pack(MAGIC, 5, 0, 0), PREFIX.pack(MAGIC, 0, 0, MAX_PAYLOAD + 1)]:
            a, b = socket.socketpair()
            receiver = Channel(b, os.urandom(32), os.urandom(32))
            try:
                a.sendall(prefix)
                with self.assertRaises(ProtocolError): receiver.recv()
            finally: a.close(); receiver.close()

    def test_tampered_authenticated_payload(self):
        import hmac, hashlib
        a, b = socket.socketpair()
        key, nonce = os.urandom(32), os.urandom(32)
        receiver = Channel(b, key, nonce, server=True)
        header, payload = pack(torch.arange(4))
        prefix = PREFIX.pack(MAGIC, 0, len(header), len(payload))
        digest = hmac.new(key, nonce + b'C' + prefix + header + payload, hashlib.sha256).digest()
        try:
            changed = payload[:-1] + bytes([payload[-1] ^ 1])
            a.sendall(prefix + digest + header + changed)
            with self.assertRaisesRegex(ProtocolError, 'authentication'): receiver.recv()
        finally: a.close(); receiver.close()


class SpanTests(unittest.TestCase):
    def test_checkpoint_identity_checks_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{}')
            (root / "model.safetensors").write_bytes(b'example immutable weights')
            record = root / 'identity-manifest.json'
            record.write_text(json.dumps(manifest(root)))
            first = verify(root, record)
            self.assertEqual(len(first), 64)
            (root / "model.safetensors").write_bytes(b'changed immutable weights')
            with self.assertRaises(ValueError): verify(root, record)

    def test_load_only_owned_modules(self):
        stage = toy_stage(1, 4)
        self.assertEqual(stage.model.loaded_keys, ["module.1", "module.2", "module.3"])
        self.assertEqual([i for _, _, i in stage.model.fwd_modules], [1, 2, 3])

    def test_reject_cache_recurrent_and_unknown_architecture(self):
        for attr, value in [("recurrent_state_cls", object), ("loaded_tp", True)]:
            m = ToyModel(); setattr(m, attr, value)
            with self.assertRaises(NotImplementedError):
                DenseStage(m, 0, 5, device="cpu", model_id="x", source_id="x")
        m = ToyModel(); m.config.architecture = "Qwen4ExpForConditionalGeneration"
        with self.assertRaises(NotImplementedError):
            DenseStage(m, 0, 5, device="cpu", model_id="x", source_id="x")
        m = ToyModel(); m.cache_weakrefs = {1: lambda: object()}
        with self.assertRaises(ValueError): DenseStage(m, 0, 5, device="cpu", model_id="x", source_id="x")

    def test_reject_cache_parameter_at_execution(self):
        s = toy_stage(0, 5)
        with self.assertRaises(ValueError):
            s.forward(dict(input_ids=torch.ones((1, 3), dtype=torch.long), activation=None,
                           last_tokens_only=1, cache="not-serializable"))

    def test_stage_forward_order_and_head_crop(self):
        ids = torch.arange(21).reshape(3, 7)
        all_stage, a, b = toy_stage(0, 5), toy_stage(0, 2), toy_stage(2, 5)
        expected = all_stage.forward(dict(input_ids=ids, activation=None, last_tokens_only=2))
        x = a.forward(dict(input_ids=ids, activation=None, last_tokens_only=2))
        got = b.forward(dict(input_ids=ids, activation=x, last_tokens_only=2))
        self.assertEqual(tuple(got.shape), (3, 2, 9))
        self.assertTrue(torch.equal(got, expected))

    def test_worker_error_aborts_connection(self):
        with worker(toy_stage(2, 5)) as (port, token, _):
            client = StageClient("127.0.0.1", port, token)
            with self.assertRaisesRegex(RuntimeError, "remote stage failed"):
                client.forward(torch.ones((1, 3), dtype=torch.long), None)

    def test_partition_and_identity(self):
        class C:
            def __init__(self, begin, end): self.manifest = toy_stage(begin, end).manifest
        Pipeline([C(0, 2), C(2, 5)])
        for stages in [[C(0, 2), C(3, 5)], [C(0, 3), C(2, 5)], [C(0, 4)]]:
            with self.assertRaises(ValueError): Pipeline(stages)
        for key in ["model_id", "source_id"]:
            a, b = C(0, 2), C(2, 5); b.manifest[key] = "different"
            with self.assertRaises(ValueError): Pipeline([a, b])

    def test_two_process_tcp_numerical_parity(self):
        token = os.urandom(32)
        children, clients = [], []
        try:
            for begin, end in [(0, 2), (2, 5)]:
                p = subprocess.Popen([sys.executable, __file__, "--child"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                children.append(p)
                p.stdin.write(json.dumps(dict(begin=begin, end=end, token=token.hex())) + "\n")
                p.stdin.flush()
                info = json.loads(p.stdout.readline())
                self.assertFalse(info["gpu_initialized"])
                clients.append(StageClient("127.0.0.1", info["port"], token, timeout=10))
            pipeline = Pipeline(clients)
            reference = toy_stage(0, 5)
            for ids in [torch.arange(12).reshape(2, 6), torch.tensor([[5]]), torch.arange(16).reshape(1, 16)]:
                expected = reference.forward(dict(input_ids=ids, activation=None, last_tokens_only=1))
                self.assertTrue(torch.equal(pipeline.forward(ids), expected))
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            for c in clients: c.close()
            for p in children:
                p.stdin.close(); p.stdin = None
                try:
                    out, err = p.communicate(timeout=10)
                    self.assertEqual(p.returncode, 0, err)
                    self.assertIn('"gpu_initialized": false', out)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(); raise


def child():
    config = json.loads(sys.stdin.readline())
    stage = toy_stage(config["begin"], config["end"])
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0)); listener.listen(1)
        print(json.dumps(dict(port=listener.getsockname()[1], gpu_initialized=torch.cuda.is_initialized())), flush=True)
        sock, _ = listener.accept()
        try: serve_connection(sock, bytes.fromhex(config["token"]), stage)
        except EOFError: pass
    print(json.dumps(dict(gpu_initialized=torch.cuda.is_initialized())), flush=True)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        unittest.main(verbosity=2)
