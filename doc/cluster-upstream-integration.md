# Experimental two-host inference

This branch adds an explicit layer-pipeline serving path on top of Kyojin
1.8.11 (`a4845bc29e91f227c73fffe1b6e5b7b7f8f58add`). Each host loads its own
contiguous layer span and owns that span's cache and recurrent state. Hidden
activations cross an authenticated TCP connection; model weights do not move
between hosts during a request. The final stage owns the output head and draft
head. A token still traverses the stages in order, so this adds model capacity
without promising twice the single-request generation speed.

The normal single-host servers do not select this path automatically. The
entry points are `tools.multinode.serve_qwen`, `tools.multinode.serve_glm`, and
`tools.multinode.serve_ds41`; the peer runs `tools.multinode.worker`.

## Scope

| Adapter | Implemented scope | Important limits |
| --- | --- | --- |
| Dense Llama/Qwen2/Qwen3 stage fixtures | Native module-span loading, cached/uncached forward, exact transport fixtures | Primarily validation/probe entry points |
| Qwen4Exp | Text, recurrent/PLE ownership, fixed session slots, greedy MTP, existing Qwen HTTP formatting | Greedy only; no distributed vision or retained prefix store |
| GLM5Next | Text, mHC/KDA/MLA state, fixed slots, greedy MTP, existing GLM HTTP formatting | Complete block splits and independent full indexers; no shared-index layout or distributed vision |
| DeepSeek V4.1 Flash | Native text serving, sampled target verification, MTP, prefill scheduling, distributed RAM prefix reuse | Current launcher targets the 40-layer Flash layout and splits target layers at 20; not a general partition planner |

Qwen/GLM reject unsupported sampling, token-suppression, logprob and other
request controls rather than treating them as implemented. Their shared adapter
must track the upstream HTTP handler's `generate()` contract, including live
token counters. DeepSeek has its own sampling path and HTTP contract.

Session operations use prepare/commit/release transactions. Cancellation drains
the in-flight transaction before releasing state. An uncertain network or model
failure poisons the pipeline; restart both stages with fresh active-session
state instead of retrying against potentially inconsistent caches. There is no
automatic worker replacement or cross-host session migration.

The cluster prefix pool is RAM-only. The upstream recurrent checkpoint disk
tier remains a separate implementation: setting `EXL3_PREFIX_DISK_DIR` does not
make the distributed DeepSeek pool persistent. Automatic model switching and
dumping/reloading the complete distributed cache are not implemented.

## Relationship to upstream optimizations

Upstream already includes the opt-in fused K6/K8 MoE, dense K8 projection and
wide MoE launch changes from PRs #104, #105 and #106. This branch uses those
implementations and retains only its additional changes; it does not duplicate
those PRs. Native source changes in this branch and upstream require a rebuilt
extension, not the pre-update binary.

Match source, native build, dependencies and numerical options on both hosts.
The source handshake hashes core source and every Python module under
`tools/multinode`, including helper files. Changing line endings changes that
fingerprint. The handshake is not a substitute for checking the native binary,
environment and model manifest.

New upstream defaults do not constitute a cluster qualification. In particular,
the standalone Generator's interleaving scheduler differs from the cluster
scheduler, the faster GDN path is specific to Qwen, and block graphs require
model-path and state-lifecycle validation. The
[configuration example](../examples/multinode/ds41-268-coordinator.json) records
the pre-update DeepSeek settings, including explicit disabled experiments. It
is a reference profile, not a universal performance preset or an installer.

## Manual DeepSeek startup

The following Linux commands illustrate the current 40-layer DeepSeek adapter
with two small 2,048-token session slots. They do not establish a maximum-context
memory budget. Qualify the checkpoint, partition, kernels and capacity before
increasing those limits. Use the same source revision and environment on both
hosts, and run commands from the repository root with the matching Python/ROCm
environment activated.

Each host needs the same complete, locally readable checkpoint on disk; only
its assigned span is loaded into memory. Shards outside a model directory via
symlinks are not supported by the current integrity verifier. Create an
integrity manifest once, then copy it unchanged to the other host:

```bash
export MODEL_DIR=/path/to/deepseek-flash-pack
export CLUSTER_STATE=/path/to/private-cluster-state
mkdir -p "$CLUSTER_STATE"
chmod 700 "$CLUSTER_STATE"
python -m tools.multinode.identity --model "$MODEL_DIR" \
  --output "$CLUSTER_STATE/integrity.json"
```

Generate a random shared token once and copy its file privately to the second
host. Both copies must belong to the service user with mode `0600`. Do not
print it or place its contents in command arguments:

```bash
umask 077
python - "$CLUSTER_STATE/token" <<'PY'
import secrets, sys
from pathlib import Path
with Path(sys.argv[1]).open('x') as stream:
    stream.write(secrets.token_hex(32) + '\n')
PY
```

Create `$CLUSTER_STATE/engine-env.json` on each host as a JSON object of string
environment values. The example profile's `environment` member is the format
expected by `--env`; the full profile object is not. Keep model-affecting
settings consistent. The recorded glibc allocator override was qualified on the
coordinator only and is a host-memory policy, not a model setting.

On the worker, load that environment before importing the model and start the
final span. `--modules 22:44` refers to the adapter's module numbering; it maps
to target layers `[20, 40)` plus the final components:

```bash
python - "$CLUSTER_STATE/engine-env.json" -m tools.multinode.worker \
  --model "$MODEL_DIR" --weights-manifest "$CLUSTER_STATE/integrity.json" \
  --token-file "$CLUSTER_STATE/token" --modules 22:44 --device cuda:0 \
  --host 127.0.0.1 --port 9034 --once --accept-timeout 1800 \
  --cache-context 2048 --cache-sessions 2 --cache-bits 16 \
  --max-rows 256 --experimental-ds41 --speculation-depth 3 <<'PY'
import json, os, sys
from pathlib import Path
settings = json.loads(Path(sys.argv[1]).read_text())
assert all(isinstance(k, str) and isinstance(v, str) for k, v in settings.items())
os.environ.update(settings)
os.execv(sys.executable, [sys.executable, *sys.argv[2:]])
PY
```

On the coordinator, keep an SSH tunnel to the worker open, using the worker's
dedicated interconnect address when available:

```bash
ssh -N -o Compression=no -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:19034:127.0.0.1:9034 USER@WORKER_ADDRESS
```

In another coordinator shell, start the first span and HTTP server:

```bash
python -m tools.multinode.serve_ds41 \
  --model "$MODEL_DIR" --weights-manifest "$CLUSTER_STATE/integrity.json" \
  --token-file "$CLUSTER_STATE/token" --env "$CLUSTER_STATE/engine-env.json" \
  --peer 127.0.0.1 --peer-port 19034 --context 2048 --sessions 2 \
  --depth 3 --draft-depth 3 --chunk 256 --max-rows 256 \
  --host 127.0.0.1 --port 8731 --model-id deepseek-flash-cluster
```

This manual path does not install a service or add an external memory watchdog.
Stop the coordinator before the worker. Cached workers require `--once` because
reconnection must create fresh state. `/health` and `/v1/models` expose the
configured context, session count, drafting and prefix settings.

Direct USB4 binding is an alternative to SSH forwarding. The worker accepts an
explicit private bind address plus `--usb4-interface thunderboltN` and validates
that the address belongs to the USB4 peer interface. The tensor protocol
authenticates frames but does not encrypt activation contents; use the SSH
tunnel where confidentiality is required. Public HTTP exposure is a separate
deployment decision and is not enabled by these commands.

## Reference harnesses and deployment wrappers

The `*_service_guard.py` modules preserve the original managed deployment and
physical-memory watchdog. They contain installation-specific paths, hostnames,
service names and interface settings. They are not used by the manual launch
above and should not be treated as generic deployment automation. The historical
`ds41_public_transfer.py` downloader is omitted from this integration branch;
it remains available in the preserved `e128717` snapshot and is not a runtime
dependency.

The reference benchmark scripts retain checkpoint-specific partitions, fixtures
and model assertions. Their network endpoint now defaults to loopback and is
selected with `--host` and `--port`. Worker-launching harnesses additionally take
`--usb4-interface` for explicit direct USB4 binds. To reproduce the original
network choice, pass `--host 10.203.0.2 --port 9032` to the Qwen/dense harnesses
(GLM used port `9034`) and add `--usb4-interface thunderbolt0` to worker roles.
For an SSH tunnel, keep the worker on loopback and set the coordinator's port
to the forwarded local port. Qwen's optional `--striped` run requires an explicit
`--aux-host` and accepts `--aux-port`; it no longer embeds a machine's link-local
address. The cached dense gate checks an unrelated primary's health only when
`--health-url` is supplied. CPU test addresses are synthetic fixtures, not
live-host dependencies.

The [pre-update snapshot](cluster-deployed-baseline-20261105.md) and
[qualification history](multinode.md) retain earlier measurements with their
original scope. Those results do not qualify a rebuilt 1.8.11 binary. Integration
validation must cover HTTP contracts, native parity, prefix restore, cancellation,
two active sessions and memory behavior using the selected deployment options.

## 1.8.11 integration validation

Both hosts ran the same integrated source fingerprint
`1aa527d64f45643bf378932a454405835c0adc6204d7b68628ce369f28557966`
and rebuilt native-extension SHA-256
`8bf2e0907212a829d4aae744c2a10f7736edabf93b9d514edea4effb39e0e7b8`.
Python, dependency versions, service units and virtual environments were
unchanged from their backups. All three saved profiles and existing environment
values were retained; graph policy was made explicit with `EXL3_HIP_GRAPH=0`
and `EXL3_BLOCK_GRAPH_MAX_Q=1`.

The final file audit verified all 1,938 tracked files and executable modes on
both hosts against the published Git tree. It caught Windows archive newline
conversion in the initial deployment; normalizing those files to the exact Git
bytes changed the source fingerprint from `ad26e768...` to the value above.
Native bytes and inference settings were unchanged. After restarting with the
canonical source, the HTTP output, MTP counts, prefix reuse, concurrent-session,
cancellation and seeded-sampling checks passed again.

Validation passed:

- 161 executions across 29 cluster CPU suites; compatibility and logging suites
  brought the total to 180 passed and four optional RigMark skips. Additional
  pytest checks passed 24 tests and five subtests.
- 528 skinny-GEMM GPU cases and 112 comparisons using actual GLM K8 weights.
  The first four DeepSeek layers passed 15 forward and seven journal/prefix
  cases exactly, using the original Engram data.
- Two full-model 128-token greedy responses retained identical output hashes
  and MTP acceptance counts before and after the update.
- Two concurrent requests completed. Cancelling an SSE request after three
  chunks was followed by two successful concurrent requests. A 64-token sampled
  response at temperature 1, top-p 0.95 and seed 812 repeated exactly.

The bounded before/after check measured:

| Workload | Before | Initial integration | Canonical-source restart |
| --- | ---: | ---: | ---: |
| Coding generation | 19.82 tokens/s | 21.03 tokens/s | 18.31 tokens/s |
| Story generation | 10.98 tokens/s | 11.62 tokens/s | 10.99 tokens/s |
| Prefill, 6,416 tokens | 536.96 tokens/s | 546.56 tokens/s | 513.25 tokens/s |
| Repeat request reusing 6,415 tokens | 0.605 s | 0.581 s | 0.713 s |

These are short regression measurements, not a stable speedup estimate
or a general response-quality benchmark. The candidate's first request also
incurred cold/JIT startup: prefill took 13.44 seconds versus 0.70 seconds for
the baseline; the canonical-source restart incurred 3.46 seconds. Warm
measurements do not remove that startup cost.

A subsequent warm repeat on canonical source measured 20.10 tokens/s for the
coding prompt and 11.00 tokens/s for the story prompt, again with exact saved
outputs and MTP counts. That is consistent with the saved baseline rather than
evidence for a substantial generation-speed change.

The managed DeepSeek service retained two 262,144-token session slots, 524,288
aggregate cache tokens, requested MTP depth 3 with capacity 5, and the 2 GiB
physical-RAM guard. The lowest observed physical available memory during these
bounded tests was approximately 24.76 GiB on the coordinator and 22.59 GiB on
the worker. This was not a maximum-context or maximum-cache-occupancy test.
Standalone Qwen was unchanged. The component checks above do not constitute
new full-model Qwen/GLM serving qualification or an extended memory soak.
