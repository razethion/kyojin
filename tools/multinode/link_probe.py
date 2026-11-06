"""Measure the real tensor round-trip path separately from model computation.

Bind loopback and tunnel SSH over USB4. Results include serialization, CPU copies,
HMAC and SSH overhead; use iperf3 separately for raw network throughput.
"""
import argparse
import ipaddress
import json
import os
import socket
import statistics
import time

from .wire import Channel, ProtocolError


def echo_connection(sock, token):
    channel = Channel.accept(sock, token)
    try:
        while True:
            message = channel.recv()
            if type(message) is not dict or set(message) != {"tensor"}:
                raise ProtocolError("expected benchmark tensor")
            channel.send(message)
    finally:
        channel.close()


def measure(host, port, token, sizes, samples, *, auxiliary=None):
    import torch
    rows = []
    if auxiliary is None:
        channel = Channel.connect(host, port, token)
    else:
        from .striped import connect_channel
        channel = connect_channel((host,port),auxiliary,token)
    try:
        for size in sizes:
            # Fixed nonzero payload. Compression should be disabled on the SSH
            # tunnel; otherwise this is not a useful link-capacity experiment.
            value = torch.arange(251, dtype=torch.uint8).repeat((size + 250) // 251)[:size].contiguous()
            timings = []
            for i in range(samples + 1):
                start = time.perf_counter()
                channel.send({"tensor": value})
                response = channel.recv()
                elapsed = time.perf_counter() - start
                if not isinstance(response, dict) or not torch.equal(response.get("tensor"), value):
                    raise ProtocolError("benchmark data corruption")
                if i:
                    timings.append(elapsed)
            med = statistics.median(timings)
            rows.append(dict(payload_bytes_each_direction=size, samples=samples,
                             median_roundtrip_ms=med * 1000,
                             combined_payload_bytes_per_second=2 * size / med,
                             minimum_roundtrip_ms=min(timings) * 1000))
    finally:
        channel.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9030)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max-mib", type=int, default=16)
    args = ap.parse_args()
    token = bytes.fromhex(os.environ.get("EXL3_NODE_TOKEN", ""))
    if len(token) < 32: ap.error("set the per-experiment EXL3_NODE_TOKEN")
    if not 1 <= args.samples <= 100 or not 1 <= args.max_mib <= 128:
        ap.error("samples must be1..100 and max-mib1..128")
    if args.server:
        if not ipaddress.ip_address(args.host).is_loopback:
            ap.error("bind loopback; tunnel over dedicated USB4 SSH")
        with socket.socket() as listener:
            listener.bind((args.host, args.port)); listener.listen(1)
            print(json.dumps({"listening": listener.getsockname()}), flush=True)
            sock, _ = listener.accept()
            try: echo_connection(sock, token)
            except EOFError: pass
    else:
        sizes = sorted({0, 1024, 64 * 1024, 1024**2, args.max_mib * 1024**2})
        print(json.dumps({"scope": "application path including serialization/authentication/tunnel",
                          "results": measure(args.host, args.port, token, sizes, args.samples)}, indent=2))


if __name__ == "__main__": main()
