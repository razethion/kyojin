"""Stage worker, loopback by default; optional validated direct USB4 binding."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from contextlib import ExitStack

from .wire import Channel, ProtocolError
from .network import validate_binding


def serve_connection(sock, token, stage):
    channel = None
    try:
        channel = Channel.accept(sock, token)
        hello = channel.recv()
        if hello != {"op": "hello", "protocol": 1}:
            raise ProtocolError("expected versioned hello")
        channel.send({"manifest": stage.manifest})
        while True:
            request = channel.recv()
            if type(request) is not dict or set(request) != {"op", "args"}:
                raise ProtocolError("invalid operation envelope")
            if request["op"] == "close" and request["args"] is None:
                channel.send({"ok": True})
                return
            if request['op'] == 'ping' and request['args'] == {}:
                channel.send({'output': {'alive': True}})
                continue
            speculative=stage.manifest.get('mode') in ('cached-qwen-spec-v1','cached-glm-spec-v1','cached-ds41-spec-v1')
            cached=stage.manifest.get('mode') in ('cached-dense-v1','cached-qwen-spec-v1','cached-glm-spec-v1','cached-ds41-spec-v1')
            allowed=('open','prepare','commit','release','draft') if speculative else ('open','prepare','commit','release')
            if stage.manifest.get('mode') == 'cached-ds41-spec-v1': allowed += ('configure_sampling','capacity','debug','idle_release','prefix_save','prefix_estimate','prefix_stats','prefix_restore','prefix_drop','prefix_clear')
            if request["op"] != "forward" and not (cached and request['op'] in allowed):
                raise ProtocolError("unsupported operation")
            try:
                output = stage.execute(request['op'],request['args']) if cached else stage.forward(request["args"])
                channel.send({"output": output})
            except Exception as e:
                # Any failed forward poisons the connection. Never retry a
                # partially executed model call against potentially stale state.
                channel.send({"error": f"{type(e).__name__}: {e}"[:1000]})
                return
    finally:
        if stage.manifest.get('mode') in ('cached-dense-v1','cached-qwen-spec-v1','cached-glm-spec-v1','cached-ds41-spec-v1'):
            stage.poisoned=True
        try:
            if stage.manifest.get('mode') == 'cached-ds41-spec-v1':
                stage.close()
        finally:
            if channel is not None: channel.close()
            else: sock.close()


class StageClient:
    def __init__(self, host, port, token, *, timeout=60, auxiliary=None):
        if auxiliary is None:
            self.channel = Channel.connect(host, port, token, timeout)
        else:
            from .striped import connect_channel
            self.channel = connect_channel((host,port),auxiliary,token,timeout)
        try:
            self.channel.send({"op": "hello", "protocol": 1})
            reply = self.channel.recv()
            if type(reply) is not dict or set(reply) != {"manifest"} or type(reply["manifest"]) is not dict:
                raise ProtocolError("invalid worker manifest")
            self.manifest = reply["manifest"]
        except BaseException:
            self.channel.close()
            raise

    def forward(self, ids, activation=None, last_tokens_only=1):
        return self.call('forward', {"input_ids": ids, "activation": activation,
                                     "last_tokens_only": last_tokens_only})

    def ping(self):
        return self.call('ping', {})

    def call(self, op, args):
        try:
            self.channel.send({"op": op, "args": args})
            reply = self.channel.recv()
            if type(reply) is not dict:
                raise ProtocolError("invalid forward reply")
            if "error" in reply:
                raise RuntimeError("remote stage failed: " + str(reply["error"]))
            if set(reply) != {"output"}:
                raise ProtocolError("invalid forward reply fields")
            return reply["output"]
        except BaseException:
            self.channel.close()
            raise

    def close(self):
        self.channel.close()


class Pipeline:
    def __init__(self, clients, *, mode='uncached-dense-v1'):
        if not clients:
            raise ValueError("no stages")
        self.clients = clients
        expected = 0
        identity = None
        for client in clients:
            m = client.manifest
            current = tuple(m.get(k) for k in ("protocol", "mode", "model_id", "source_id", "total_modules"))
            if current[0:2] != (1, mode) or not all(current[2:]):
                raise ValueError("unsupported or incomplete stage identity")
            if identity is None:
                identity = current
            if current != identity or type(m.get("begin")) is not int or type(m.get("end")) is not int or m["begin"] != expected or not expected < m["end"] <= m["total_modules"]:
                raise ValueError("stage source/model mismatch or noncontiguous partition")
            expected = m["end"]
        if len({(client.manifest.get('architecture'),client.manifest.get('hc_mult')) for client in clients})!=1:
            raise ValueError('stage architecture or residual stream geometry differs')
        if expected != identity[-1]:
            raise ValueError("partition does not cover the complete model")

    def forward(self, input_ids, last_tokens_only=1):
        x = None
        try:
            for client in self.clients:
                x = client.forward(input_ids, x, last_tokens_only)
            return x
        except BaseException:
            self.close()
            raise

    def close(self):
        for client in self.clients:
            client.close()


class CachedPipeline(Pipeline):
    """Fixed-page sessions with all-stage prepare before any committed advance.

    A partial commit/network error closes the whole pipeline. No retry or
    session continuation is allowed against unknown distributed state.
    """
    def __init__(self, clients):
        super().__init__(clients,mode='cached-dense-v1')
        from .session import Sessions
        geometry={(client.manifest.get('context'),client.manifest.get('sessions'),client.manifest.get('cache_bits')) for client in clients}
        if len(geometry)!=1:raise ValueError('stage cache geometry differs')
        context,slots,_=geometry.pop()
        self.sessions=Sessions(context,slots)
        self.failed=False

    def _require_live(self):
        if self.failed:raise RuntimeError('pipeline is closed/poisoned; restart required')

    def open(self, name):
        self._require_live()
        try:
            expected=self.sessions.open(name)
            for client in self.clients:
                if client.call('open',{'session':name})!=expected:
                    raise ProtocolError('session allocation acknowledgement differs')
        except BaseException:self.close();raise

    def forward(self, name, input_ids, last_tokens_only=1):
        import torch
        self._require_live()
        try:
            if not isinstance(input_ids,torch.Tensor) or input_ids.dtype!=torch.int64 or input_ids.ndim!=2 or input_ids.shape[0]!=1:
                raise ValueError('expected one int64 sequence')
            state=self.sessions.items[name]
            step,past=state.step,state.length
            self.sessions.begin(name,step,past,input_ids.shape[1])
            x=None
            for client in self.clients:
                x=client.call('prepare',{'session':name,'step':step,'past':past,'input_ids':input_ids,
                                        'activation':x,'last_tokens_only':last_tokens_only})
            expected={'length':past+input_ids.shape[1],'step':step+1}
            for client in self.clients:
                if client.call('commit',{'session':name,'step':step})!=expected:
                    raise ProtocolError('commit acknowledgement differs')
            self.sessions.mark_ready(name)
            self.sessions.commit(name,step)
            return x
        except BaseException:self.close();raise

    def release(self, name):
        self._require_live()
        try:
            for client in self.clients:
                if client.call('release',{'session':name})!={'released':True}:
                    raise ProtocolError('release acknowledgement differs')
            self.sessions.release(name)
        except BaseException:self.close();raise

    def close(self):
        self.failed=True
        super().close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights-manifest", required=True, help="verified identical checkpoint manifest on both nodes")
    ap.add_argument("--modules", required=True, help="half-open top-level module range, e.g. 0:18; NOT layer numbers")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9020)
    ap.add_argument("--max-rows", type=int, default=2048)
    ap.add_argument("--usb4-interface", help="explicit direct USB4 binding instead of a loopback SSH tunnel")
    ap.add_argument("--once", action="store_true", help="exit after one connection (bounded experiment worker)")
    ap.add_argument("--accept-timeout", type=float, default=120)
    ap.add_argument("--token-file", help="private user-owned experiment token file, alternative to EXL3_NODE_TOKEN")
    ap.add_argument("--cache-context", type=int, default=0, help="experimental fixed-page dense context per session")
    ap.add_argument("--cache-sessions", type=int, default=2)
    ap.add_argument("--aux-host", help="optional second USB4 link for large-frame striping")
    ap.add_argument("--aux-port", type=int, default=9033)
    ap.add_argument("--aux-interface")
    ap.add_argument("--experimental-qwen",action="store_true",help="opt-in Qwen4Exp recurrent/PLE span (no MTP/vision)")
    ap.add_argument('--experimental-glm',action='store_true',help='opt-in GLM5Next mHC/KDA/MLA text stages')
    ap.add_argument('--experimental-ds41',action='store_true',help='native DeepSeekV4.1 source-aligned spans and DSpark')
    ap.add_argument('--speculation-depth',type=int,default=0,help='experimental greedy Qwen/GLM MTP, depth1..7')
    ap.add_argument("--cache-bits",type=int,choices=(8,16),default=16)
    args = ap.parse_args()
    if args.cache_context and not args.once:
        ap.error('experimental cached workers require --once; reconnect needs fresh state')
    try:
        validate_binding(args.host, args.usb4_interface)
        if args.aux_host:
            if not args.usb4_interface or not args.aux_interface or args.usb4_interface==args.aux_interface:
                raise ValueError('striping requires two distinct explicit USB4 interfaces')
            validate_binding(args.aux_host,args.aux_interface)
        elif args.aux_interface:
            raise ValueError('auxiliary interface requires its bind address')
    except ValueError as error:
        ap.error(str(error))
    encoded=os.environ.get("EXL3_NODE_TOKEN", "")
    if args.token_file:
        path=Path(args.token_file)
        stat=path.stat()
        if stat.st_uid!=os.getuid() or stat.st_mode & 0o077:
            ap.error('experiment token file must be private and user-owned')
        encoded=path.read_text().strip()
    token = bytes.fromhex(encoded)
    if len(token) < 32:
        ap.error("EXL3_NODE_TOKEN must contain at least 32 random bytes encoded as hex")
    begin, end = map(int, args.modules.split(":"))
    if args.experimental_ds41:
        if args.experimental_qwen or args.experimental_glm or args.cache_bits != 16 or not args.cache_context:
            ap.error('DeepSeek uses its native FP16 fake-quant cache and explicit context')
        from .ds41_stage import load_stage
        first, last = (0 if begin == 0 else begin - 2), (40 if end == 44 else end - 2)
        stage = load_stage(args.model, args.weights_manifest, first, last, device=args.device,
                           context=args.cache_context, sessions=args.cache_sessions,
                           depth=args.speculation_depth, max_rows=args.max_rows)
    else:
        from .stage import load_stage
        stage = load_stage(args.model, begin, end, args.device, args.weights_manifest, args.max_rows,
                           cache_context=args.cache_context,cache_sessions=args.cache_sessions,cache_bits=args.cache_bits,
                           experimental_qwen=args.experimental_qwen,experimental_glm=args.experimental_glm,speculation_depth=args.speculation_depth)
    family = socket.AF_INET6 if ':' in args.host else socket.AF_INET
    with ExitStack() as stack:
        listener=stack.enter_context(socket.socket(family))
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(args.accept_timeout)
        address=socket.getaddrinfo(args.host,args.port,family,socket.SOCK_STREAM)[0][4]
        listener.bind(address); listener.listen(1)
        auxiliary=None
        if args.aux_host:
            auxiliary_family=socket.AF_INET6 if ':' in args.aux_host else socket.AF_INET
            auxiliary=stack.enter_context(socket.socket(auxiliary_family))
            auxiliary.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            auxiliary.settimeout(args.accept_timeout)
            auxiliary.bind(socket.getaddrinfo(args.aux_host,args.aux_port,auxiliary_family,socket.SOCK_STREAM)[0][4])
            auxiliary.listen(1)
        print(json.dumps({"listening": listener.getsockname(), "manifest": stage.manifest}), flush=True)
        while True:
            sock, _ = listener.accept()
            try:
                if auxiliary:
                    from .striped import accept_socket
                    sock=accept_socket(sock,auxiliary,token,timeout=args.accept_timeout)
                serve_connection(sock, token, stage)
            except (OSError, EOFError, ProtocolError) as e:
                print(f"connection closed: {type(e).__name__}", flush=True)
            if args.once:
                break


if __name__ == "__main__":
    main()
