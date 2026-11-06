"""Bounded, authenticated tensor frames. No pickle, code or object handles on wire.

Listeners must remain on loopback; use SSH forwarding over the dedicated USB4
interface. HMAC authenticates frames but does not itself encrypt activations.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import socket
import struct
import sys

MAGIC = b"EXL3MN01"
PREFIX = struct.Struct("!8sQIQ")
MAX_HEADER = 64 * 1024
MAX_PAYLOAD = 256 * 1024**2
MAX_NODES = 8192
DTYPES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
          "int64": 8, "int32": 4, "uint8": 1, "bool": 1}


class ProtocolError(RuntimeError):
    pass


def exact(sock, size):
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset < size:
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise EOFError("peer disconnected during frame")
        offset += n
    return result


def pack(value, max_payload=MAX_PAYLOAD):
    import torch
    if sys.byteorder != "little":
        raise ProtocolError("only little-endian tensor hosts are supported")
    tensors, chunks = [], []
    total = nodes = 0

    def visit(x, depth=0):
        nonlocal total, nodes
        nodes += 1
        if depth > 20 or nodes > MAX_NODES:
            raise ProtocolError("message tree exceeds limits")
        if isinstance(x, torch.Tensor):
            dtype = str(x.dtype).removeprefix("torch.")
            if dtype not in DTYPES or x.layout != torch.strided or x.ndim > 8:
                raise ProtocolError("unsupported tensor dtype/layout/rank")
            size = x.numel() * x.element_size()
            if total + size > max_payload:
                raise ProtocolError("tensor payload exceeds limit")
            raw = x.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            index = len(tensors)
            tensors.append(dict(dtype=dtype, shape=list(x.shape), offset=total, size=size))
            chunks.append(raw)
            total += size
            return ["tensor", index]
        if x is None or type(x) in (str, bool, int, float):
            return ["scalar", x]
        if type(x) is dict:
            if any(type(k) is not str for k in x):
                raise ProtocolError("dictionary keys must be strings")
            return ["dict", [[k, visit(v, depth + 1)] for k, v in x.items()]]
        if type(x) in (list, tuple):
            return ["list" if type(x) is list else "tuple", [visit(v, depth + 1) for v in x]]
        raise ProtocolError(f"unsupported wire object: {type(x).__name__}")

    tree = visit(value)
    header = json.dumps(dict(tree=tree, tensors=tensors), allow_nan=False,
                        separators=(",", ":")).encode()
    if len(header) > MAX_HEADER:
        raise ProtocolError("metadata exceeds limit")
    return header, b"".join(chunks)


def unpack(header, payload):
    import torch
    if len(header) > MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ProtocolError("frame exceeds limits")
    try:
        meta = json.loads(header, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if set(meta) != {"tree", "tensors"} or type(meta["tensors"]) is not list:
            raise ValueError("invalid metadata")
        if len(meta["tensors"]) > MAX_NODES:
            raise ValueError("too many tensors")
        arrays = []
        cursor = 0
        # Validate the entire tensor layout before allocating any torch storage.
        for t in meta["tensors"]:
            if set(t) != {"dtype", "shape", "offset", "size"}:
                raise ValueError("invalid tensor fields")
            shape = t["shape"]
            if type(shape) is not list or len(shape) > 8 or any(type(n) is not int or n < 0 or n > MAX_PAYLOAD for n in shape):
                raise ValueError("invalid tensor shape")
            expected = math.prod(shape) * DTYPES[t["dtype"]]
            if type(t["size"]) is not int or type(t["offset"]) is not int or t["offset"] != cursor or t["size"] != expected:
                raise ValueError("invalid tensor size or offset")
            cursor += expected
            if cursor > len(payload):
                raise ValueError("truncated tensor payload")
        if cursor != len(payload):
            raise ValueError("unclaimed payload bytes")
        for t in meta["tensors"]:
            dtype = getattr(torch, t["dtype"])
            if t["size"]:
                raw = bytearray(payload[t["offset"]:t["offset"] + t["size"]])
                array = torch.frombuffer(raw, dtype=dtype).reshape(t["shape"])
            else:
                array = torch.empty(t["shape"], dtype=dtype)
            arrays.append(array)
        nodes = 0

        def visit(node, depth=0):
            nonlocal nodes
            nodes += 1
            if depth > 20 or nodes > MAX_NODES or type(node) is not list or len(node) != 2:
                raise ValueError("invalid message tree")
            tag, value = node
            if tag == "tensor" and type(value) is int and 0 <= value < len(arrays):
                return arrays[value]
            if tag == "scalar" and (value is None or type(value) in (str, int, float, bool)):
                return value
            if tag in ("list", "tuple") and type(value) is list:
                items = [visit(x, depth + 1) for x in value]
                return tuple(items) if tag == "tuple" else items
            if tag == "dict" and type(value) is list:
                out = {}
                for pair in value:
                    if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in out:
                        raise ValueError("invalid dictionary")
                    out[pair[0]] = visit(pair[1], depth + 1)
                return out
            raise ValueError("invalid message tag")
        return visit(meta["tree"])
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError, RuntimeError) as e:
        raise ProtocolError(f"invalid tensor message: {e}") from e


class Channel:
    def __init__(self, sock, token, nonce, server=False):
        if len(token) < 32:
            raise ValueError("use at least 32 random bytes for the experiment token")
        self.sock, self.token, self.nonce = sock, token, nonce
        # Each tensor frame has a small authenticated header followed by its
        # payload. Nagle plus delayed ACK can stall small decode frames ~40ms.
        # Keep this off for TCP; socketpair-based protocol tests also use Unix
        # sockets, where TCP options do not apply.
        if getattr(sock, 'family', None) in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.tx_direction, self.rx_direction = (b"S", b"C") if server else (b"C", b"S")
        self.tx = self.rx = 0

    @classmethod
    def accept(cls, sock, token, timeout=60):
        sock.settimeout(timeout)
        nonce = os.urandom(32)
        sock.sendall(MAGIC + nonce)
        return cls(sock, token, nonce, server=True)

    @classmethod
    def connect(cls, host, port, token, timeout=60):
        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            if bytes(exact(sock, len(MAGIC))) != MAGIC:
                raise ProtocolError("unsupported protocol")
            return cls(sock, token, bytes(exact(sock, 32)))
        except BaseException:
            sock.close()
            raise

    def send(self, message):
        header, payload = pack(message)
        prefix = PREFIX.pack(MAGIC, self.tx, len(header), len(payload))
        digest = hmac.new(self.token, self.nonce + self.tx_direction + prefix, hashlib.sha256)
        digest.update(header); digest.update(payload)
        self.sock.sendall(prefix + digest.digest() + header)
        self.sock.sendall(payload)
        self.tx += 1

    def recv(self):
        prefix = bytes(exact(self.sock, PREFIX.size))
        magic, seq, hlen, plen = PREFIX.unpack(prefix)
        if magic != MAGIC or seq != self.rx or hlen > MAX_HEADER or plen > MAX_PAYLOAD:
            raise ProtocolError("invalid frame version, sequence or length")
        signature = bytes(exact(self.sock, 32))
        header, payload = exact(self.sock, hlen), exact(self.sock, plen)
        digest = hmac.new(self.token, self.nonce + self.rx_direction + prefix, hashlib.sha256)
        digest.update(header); digest.update(payload)
        if not hmac.compare_digest(signature, digest.digest()):
            raise ProtocolError("frame authentication failed")
        result = unpack(header, payload)
        self.rx += 1
        return result

    def close(self):
        self.sock.close()
