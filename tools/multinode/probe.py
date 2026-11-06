"""Uncached dense-text integration probe, not an OpenClaw serving replacement."""
import argparse
import json
import os

from .identity import verify
from .worker import Pipeline, StageClient


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local tokenizer and verified checkpoint directory")
    ap.add_argument("--weights-manifest", required=True)
    ap.add_argument("--stage", action="append", required=True, help="host:port in module execution order")
    ap.add_argument("--prompt", required=True, help="raw prompt; no implicit chat template")
    ap.add_argument("--tokens", type=int, default=8)
    args = ap.parse_args()
    if not 1 <= args.tokens <= 64:
        ap.error("pilot completion budget is 1..64 tokens")
    token = bytes.fromhex(os.environ.get("EXL3_NODE_TOKEN", ""))
    if len(token) < 32:
        ap.error("set the per-experiment EXL3_NODE_TOKEN")
    identity = verify(args.model, args.weights_manifest)
    from exllamav3 import Config, Tokenizer
    tokenizer = Tokenizer.from_config(Config.from_directory(args.model))
    ids = tokenizer.encode(args.prompt)
    clients = []
    try:
        for address in args.stage:
            host, port = address.rsplit(":", 1)
            client = StageClient(host, int(port), token)
            clients.append(client)
            if client.manifest["model_id"] != identity:
                raise ValueError("worker checkpoint differs from coordinator tokenizer/checkpoint")
        pipeline = Pipeline(clients)
        generated = []
        import torch
        for _ in range(args.tokens):
            logits = pipeline.forward(ids)
            next_id = logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(int(next_id.item()))
            ids = torch.cat([ids, next_id], dim=1)
        print(json.dumps(dict(token_ids=generated, text=tokenizer.decode(torch.tensor([generated])))))
    finally:
        for client in clients: client.close()


if __name__ == "__main__": main()
