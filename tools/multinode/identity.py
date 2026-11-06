"""Local checkpoint integrity manifest; paths are never supplied by network peers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

METADATA = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer.model",
            "tokenizer_config.json", "generation_config.json", "special_tokens_map.json",
            "quantization_config.json", "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja")


def files(root):
    root = Path(root).resolve()
    paths = sorted(set(root.glob("*.safetensors")) | set(root.glob('*.st')) | {root / name for name in METADATA if (root / name).is_file()})
    if not (root / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in paths):
        raise ValueError("checkpoint needs config.json and safetensors weights")
    for p in paths:
        if p.resolve().parent != root:
            raise ValueError("pilot requires checkpoint files local to the model directory")
    return paths


def digest_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(8 * 1024**2):
            sha.update(data)
    return sha.hexdigest()


def manifest(root):
    return dict(version=1, files=[dict(name=p.name, size=p.stat().st_size, sha256=digest_file(p)) for p in files(root)])


def verify(root, manifest_path):
    expected = json.loads(Path(manifest_path).read_text())
    actual = manifest(root)
    if actual != expected:
        raise ValueError("checkpoint bytes or file set differ from the integrity manifest")
    return hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest(args.model), stream, indent=2)
    print("Checkpoint manifest created; no weights converted or changed.")


if __name__ == "__main__": main()
