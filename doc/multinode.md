# Multi-node inference: deployed snapshot and qualification history

## Current deployment, November 5, 2026

The deployed service is the two-node DeepSeek V4.1 Flash 1017 EXL3 2.68 bpw
text cluster. It supports sampled MTP, defaults to temperature 1.0 and top-p
0.95, and includes a repetition guard, worker-side sampling, prompt-prefix
reuse, cancellation and slot reuse. It advertises two 262,144-token sessions
and 524,288 total active cache tokens. These limits describe the configured
capacity; they are not a full-context stress-test result.

The cluster is an explicit `tools.multinode.serve_ds41` deployment. Qwen and
GLM adapters and earlier test results remain in this snapshot. A gateway that
loads or switches models on demand is **not implemented**; selecting an API
model name does not start another model or change the running service.

See [the November 5 deployed baseline](cluster-deployed-baseline-20261105.md)
for audited source/native identities, the coordinator-only allocator setting,
profile scope and validation limits. The scripts named `*_service_guard.py`
are host-specific deployment wrappers, with fixed paths, service names and
network settings. They require reviewed local configuration and credentials.
They are not a portable installer.

## Historical qualification record: November 1-2, 2026

The remainder records the stages of the original dense, Qwen and GLM work.
Statements such as "current", "unsupported", "greedy only" and "primary
restored" below describe that stage of the experiment, not the November 5
DeepSeek deployment. The recorded measurements and limitations are retained;
no GPU tests were run to prepare this publication.

The implementation under `tools/multinode/` is an experimental layer pipeline
with node-local weights and cache state. Ordinary Kyojin servers do not select
it. A GLM text pilot is now served to the existing OpenClaw GLM entry;
this remains an experimental distributed server. See the serving section below.

Native dense GPU stages now pass exact parity, including two EVO X2 machines
over USB4. The original nonfinite-output failure was an FP16 skinny-GEMM launch
bug: padded 9–32-row dispatch used a grid for a different column layout. The
isolated runtime fixes that dispatch and passes 528 kernel regression cases.
Production binaries were not replaced.

Qwen4Exp hybrid text has a separate opt-in adapter. The current Qwen 6.05
checkpoint now passes full-model two-host cached greedy comparison: all 384
measured output tokens and 24 sampled full logit rows match the solo reference
exactly. The separate opt-in greedy MTP path also passes the full-model
two-host test described below. Long contexts and sampled decoding remain unvalidated.

## Request and HTTP lifecycle, November 2

`tools/multinode/generation.py` adds cooperative greedy requests with chunked
prefill, two fixed session slots, EOS/output-budget termination and cancellation.
One lock covers each complete distributed prepare/commit transaction. A client
disconnect signals cancellation; pages are released only after that transaction
finishes. Invalid input and capacity rejection happen before worker mutation.
Uncertain network/commit failures still require restarting the whole pipeline.

The full 6.05 checkpoint passes 12 lifecycle records on solo and both hosts:
two interleaved requests exactly match sequential references, a 2032-token
prompt works within the 2048-token context, and cancellation during prefill or
after generation releases state. Reusing each slot reproduces the original
tokens. Single-token prompts and EOS inside an accepted draft block also pass.
First/final-node peak Torch allocations are 49.92/52.69 GB with two slots.
This is cooperative request scheduling, not simultaneous stage execution.

`tools.multinode.serve_qwen` is the opt-in HTTP entry point. It reuses Kyojin's
Qwen chat template, tool parser and OpenAI-compatible handlers. It binds to
loopback and defaults to greedy text MTP; sampled requests are rejected.
`qwen_engine.py` shields in-flight transaction threads during async cancellation,
holds partial stop strings/UTF-8 tails, and reports no surviving prefix cache.
Nine lifecycle, six async-engine and four real HTTP CPU tests pass. The actual
two-host HTTP GPU gate also passes: JSON/SSE replies match decoded solo token
references, two simultaneous requests match both references, native-template
chat returns tokens, sampled requests return400, and disconnect/reuse is exact.
This entry point does not edit or replace the primary service or OpenClaw.
The tests ended and the unchanged primary was verified responding at
262144/chat,400128shared cache,two sessions,MTP7. Production-size caches,
sampled decoding, prefix reuse and GLM still require separate work.

## Distributed MTP gate, November 2, 2026

| Same adapter and settings | Aggregate generation | Per-prompt range |
|---|---:|---:|
| Solo MTP on llama2 | 38.37 tokens/s | 36.28–40.05 |
| Two-host MTP, 24 blocks each | 28.94 tokens/s | 27.26–30.24 |

Both arms use current 6.05, streamed NG6, calibrated mixers, native K6 fusion,
Q8 cache, 2048 tokens, one active session and the native depth-7 tier policy.
Timing excludes every token in the first emitted batch, avoiding first-chunk
MTP bias. Clustering costs 24.58% throughput in this short test. Earlier
target-only clustering measured 17.74 tokens/s; that earlier path also returned
full logits, whereas the MTP path scores and samples on the final node.

All **528 checked output tokens**, every proposal/verification/accepted-prefix
round, and **43 full logit rows** match solo MTP exactly. The speed runs accept
232 of 339 drafted tokens (68.44%) in both arms. Thirteen native recurrent
rollback cases cover all rejected suffix lengths 0–7 and a page crossing;
seven CPU transaction tests cover prefix commit, replay, failure and real TCP.
Peak Torch allocations: 49.42 GB first node and 52.26 GB final node, versus
101.44 GB for the whole solo model with MTP. These are allocations, not total RAM.

The native MTP generator itself differs from native plain greedy text on two
of these three prompts. A saved-prefix first-token R1/R2 probe was exact at
every module, so it does not explain the later divergence. The cluster preserves
the tested native MTP behavior; these results do not establish general bitwise
equivalence between MTP and plain R1 decoding. No weights were changed.

Enable only with `--experimental-qwen --cache-context 2048 --cache-bits 8
--cache-sessions 1 --speculation-depth 7 --once`. The coordinator uses
`SpeculativePipeline`, not the uncached example below. Target hidden states,
the native MTP model and shared output head remain on the final node; only its
CPU embedding is duplicated. Verification records local recurrent history,
then both nodes commit the same accepted prefix using native advance/rewind.
An uncertain commit poisons the pipeline and requires fresh workers.

The benchmark is `tests/qwen_cluster_mtp_bench.py`. Detailed run receipts are
retained outside this public source snapshot.
The primary was restored and verified responding with MTP7, two sessions,
262144 tokens/chat and 400128 shared cache tokens. Temporary tokens/listeners
were removed; both SSH control sessions remain open.

## Qwen 6.05 measurement, November 1, 2026

| Execution | Generation tokens/s | Scope |
|---|---:|---|
| Solo on llama2 | 21.21 aggregate | Target only, MTP off |
| 24 blocks on each host | 17.74 aggregate | Same weights/settings, MTP off |
| Unchanged primary on llama | 36.11–38.83 | Production MTP7, separate API baseline |

The matched arms use streamed NG6, calibrated group-scale mixers, native K6
fusion, Q8 cache, one session and **2048 cache tokens**, with a 24-token warmup
and three 128-token greedy generations. Aggregate is total measured decode
tokens divided by total decode time. The clustered arm is 16.37% slower, or
19.58% more time per token. It uses the rear USB4 link, not dual striping.
The primary API baseline uses its full 400128-token cache and 262144-token chat
limit; MTP event batching affects stream-visible timing. These are not a direct
MTP-enabled solo/cluster comparison.

The first-stage peak Torch allocation was 49.03 GB, versus 98.11 GB for the
complete solo reference. A required Qwen boundary fix flushes a deferred prefill
residual update before sending its FP32 streams; otherwise optimized prefill
silently loses local state and produces different logits. Four actual boundary
flushes were recorded in the accepted run. The original primary configuration
is restored after testing; no cluster serving endpoint replaces OpenClaw.

Detailed receipts, failed attempts and recovery records are retained outside
this public source snapshot.

## Implemented

- Authenticated, versioned, length-bounded TCP messages, with fresh connection
  nonces and directional sequence counters. Raw tensor bytes preserve dtype,
  including FP32 stream stacks and BF16. No pickle/object/code deserialization.
- Checkpoint manifests verify actual safetensors/config/tokenizer bytes on each
  node. The worker chooses its model path at startup; clients cannot load paths,
  run commands or select arbitrary functions. Source fingerprints must match.
- `DenseStage` selects a half-open range of top-level modules **before** loading
  weights. That includes embeddings, blocks, norms and heads, not only numbered
  transformer layers. Unowned weights are not loaded.
- Ordered stage execution and bounded final-head output. The coordinator can
  sample greedily from the final stage's logits without sending all prompt logits.
- Connection failures and worker exceptions abort the pipeline; no blind retry
  against potentially partially modified state.
- Cached sessions use fixed disjoint pages and explicit prepare/commit/release.
  Each node allocates only its owned layers' KV/recurrent buffers. An uncertain
  write poisons the worker; restarting requires fresh state, not replay.
- Optional dual-link striping preserves the authenticated logical stream over
  two explicitly validated USB4 interfaces. It is off by default. The two cables
  delivered about 19.3 Gbit/s aggregate iperf; actual 16–32 MiB tensor echo tests
  improved only about 5–8%, including copies and authentication.
- A link probe measures activation round-trip costs with fixed synthetic data.
  It includes copies/serialization/authentication/tunnel costs. Raw `iperf3` and
  application latency are separate measurements.

The default architecture allowlist is dense Llama/Qwen2/Qwen3. The explicit
`--experimental-qwen` flag enables cached Qwen4Exp text only: FP32 four-stream
activations, GDN/PLE state owned locally, original IDs on each node, position
advancement only after commit. Qwen requires an explicit cached configuration.
The additional `--speculation-depth` opt-in supports greedy Qwen MTP only.
GLM, sampled MTP, relayering and vision remain unsupported. Construct stages
before attaching a cache; loading a full model and then pruning is not supported.

## Local CPU proof

From the repository root, with CPU torch and numpy installed:

```bash
PYTHONPATH=. python tests/test_multinode_cpu.py
```

The tests use synthetic modules implementing the EXL3 module interface, not a
real quantized checkpoint. They verify the loader's range/order behavior, exact
two-process TCP numerical parity, tensor layouts/dtypes, 16 MiB transfers,
identity rejection, malformed packets, wrong tokens, partial reads, disconnects,
and worker errors. They assert no GPU context was initialized. Passing this is
not evidence of trained quantized-model correctness or representative speed.

Additional CPU tests are `tests/test_multinode_sessions_cpu.py` and
`tests/test_multinode_striped_cpu.py`. Native GPU checks include
`tests/test_multinode_skinny_gpu.py`, `tests/test_multinode_gpu_stage.py` and
`tests/test_multinode_cached_gpu.py`. The two-host dense cached gate
`tests/multinode_cached_two_host.py` passed 30 exact steps with two interleaved
sessions, a page crossing and slot reuse, on both single and dual transport.
Fixtures are tiny random FP16 weights, not useful language models.

`tests/qwen_cluster_bench.py` measures matched target-only Qwen solo/clustered
generation, separately from the primary's MTP baseline. It verifies checkpoint
bytes, generated token IDs and sampled full logit rows. Experiment budgets
and raw receipts are retained outside this public source snapshot.

## Hardware experiment prerequisites

1. Prepare and audit both hosts, then measure their USB4 networking.
2. Match source trees and ROCm/native extension builds. A source Git revision is
   insufficient if either tree has local modifications.
3. Choose a small supported dense EXL3 pack that fits on one GPU, with room for
   the reference and working memory. Do not start with a >128 GB model.
4. Inspect the model's top-level `model.modules` keys and choose a complete,
   nonoverlapping partition. Never assume module numbers equal layer numbers.
5. Build an integrity manifest once and copy it unchanged to the other node:

   ```bash
   python -m tools.multinode.identity --model /path/to/small-pack --output /path/to/pack-manifest.json
   ```

   Workers rehash the checkpoint at startup. For this first gate, the complete
   checkpoint must be available locally on each node's disk; only the assigned
   module span is loaded into RAM. Partial shard downloads are a later loader
   improvement, not yet implemented. Pilot manifests require ordinary local files,
   not safetensors symlinks out to another directory.

## Test-only launch sketch

These commands are templates, **not commands already run on llama/llama2**.
Supply an identical random per-experiment `EXL3_NODE_TOKEN` environment value to
both workers and coordinator. It is a local session authentication secret, not
an OpenAI/HF/API credential. Do not put it in Git, process arguments or logs.

On each node, activate its matched ROCm environment and start its assigned range:

```bash
export EXL3_KVWIN=0  # disable undeveloped distributed KVWIN lifecycle
python -m tools.multinode.worker --model /path/to/small-pack \
  --weights-manifest /path/to/pack-manifest.json \
  --modules START:END --device cuda:0 --port 9020 --max-rows 2048
```

Keep workers bound to loopback. Forward the second node's worker over the USB4
IP, with compression disabled:

```bash
ssh -N -o Compression=no -L 127.0.0.1:9021:127.0.0.1:9020 razeth@LLAMA2_USB4_IP
```

The coordinator runs on llama, with its own first worker at9020 and the second
worker's tunnel at9021:

```bash
python -m tools.multinode.probe --model /path/to/small-pack \
  --weights-manifest /path/to/pack-manifest.json \
  --stage 127.0.0.1:9020 --stage 127.0.0.1:9021 \
  --prompt 'A small correctness probe:' --tokens 8
```

This recomputes the full prefix each token. It exists to establish module/weight
partition correctness, not useful serving performance. Compare full logits and
greedy tokens to a local reference under identical kernels and precision. Do not
interpret matching sampled prose as a correctness proof.

An opt-in intermediate GPU gate is `PYTHONPATH=. python
tests/test_multinode_gpu_stage.py`. It generates a tiny random FP16 Llama fixture
locally and compares standard EXL3 forward to several module splits with the
byte-preserving CPU boundary. This tests native FP16 kernels on one GPU; it
does not replace quantized-checkpoint, cached decode or two-node validation.
Those original 16-row failures led to the skinny-GEMM dispatch fix described
above. The corrected fixture passes exact native stage tests without tracing;
two-host uncached and cached GPU gates also pass. Historical failure logs,
source identities and receipts are retained outside this public snapshot.

## Required work before the user's larger Qwen/GLM models

- The fixed-page cached text path and committed-boundary request cancellation
  exist. Dynamic allocation, prefix sharing, eviction and prefix save/restore
  still need work.
- Qwen's opt-in recurrent slot/commit and exact stream/PLE handling pass the
  short trained-checkpoint gate, including speculative prefix rollback.
  GLM5Next now has an explicit experimental adapter for complete block splits
  with independent full DSA indexers. Shared-index layouts remain unsupported.
- Greedy MTP state exports, head placement, acceptance and all-stage rollback
  pass the short full-model gate. An opt-in Kyojin HTTP adapter exists. Sampled
  proposal-law handling, production deployment and distributed prefix reuse
  still need work.
- Vision, stage pipelining, measured long-context quality and throughput.
  Both tiny dense and full Qwen fixtures support two interleaved sessions;
  this does not prove sustained production concurrency. Budget for duplicated
  heads/embeddings where needed.

Expect larger capacity first. One decode stream traverses the stages in order;
it does not automatically gain the sum of both memory bandwidths or twice the
generation rate. USB4 payload and synchronization costs must be measured.

## GLM5Next opt-in gate

`--experimental-glm` enables cached GLM5Next text spans. The wire preserves the
FP32 four-stream mHC activation and materializes pending residual updates at
each stage boundary. Each node owns its KDA history and MLA/indexer/pool cache.
The public GLM5.3 tiny fixture passes28 exact native/stage chunk comparisons
across FP16/Q8 caches, sparse-layer boundaries, page crossing and slot reuse.
Its64-wide KDA heads cannot use the native single-token recurrent kernel.

Production-shape validation instead uses unchanged turboderp sc_4.10bpw weights
at revision `c6b8ac3f6069173343e253e1cba90536ed2c72dc`: first eight blocks,
KDA128, MLA512, Q8 cache. Fifteen native/adapter tests pass exactly for decode,
three-row verification with one/two/three accepted inputs, and subsequent
single-token decode. Prefixes17/250/2050/2051/2052 cover page boundaries and
DSA selection activation; both full-indexer complete-pool caches also match.
These tests compare with native rewind under the same schedule; they do not
independently prove native rewind mathematics or end-to-end model quality.

`--speculation-depth 2` also enables the experimental GLM MTP adapter. It uses
GLM's native hidden-state export tap and fixed-depth draft predictions, with
its embedding on CPU and head shared on the final node. It does not assume
Qwen's Tier policy or require a Qwen draft2s probability head. Enabling this
flag alone is not a parity guarantee; the qualification scope follows below.

The full4.10 pack is167,814,068,151bytes. Both disks currently hold the complete
verified pack; only the assigned span is loaded. The two-host test split is
modules0:26 and26:50 (24 and21 transformer blocks),2048 tokens per session,
two slots, Q8 cache. `tests/glm_cluster_bench.py` measures short greedy outputs
with thinking immediately closed; it is a smoke/speed test, not LiveBench or
a formal quality score. Qwen production files and runtime remain separate.

The initial full-model target-only smoke completed three96-token continuations
at4.438/4.434/4.425 tok/s, excluding the16-token warmup. Original-node minimum
MemAvailable was42.7GB; the second node retained44.5GB. These are the initial
unoptimized engine settings with per-block graphs disabled, not a tuned GLM
throughput claim. Startup checksum verification and weight loading are excluded
from generation rate. The entire checkpoint cannot fit on one128GB node, so
these numbers have no full-model solo comparison.

The initial, unoptimized fixed-depth2 MTP run completed at6.052/7.939/7.274 tok/s (aggregate
6.999, about58% above target-only). Its proposals match verifier predictions
135/300 times. These three short continuations are not a formal quality test.
The MTP component's final-stage embedding/head binding passes four exact
native-owner comparisons for hidden states, full logits and argmax tokens.
Full-model MTP versus plain-greedy outputs nevertheless diverge at zero-based
tokens6 and8 on the first two prompts; all96 tokens match on the third.
This is not a claim of lossless MTP or full native-Generator schedule parity.
Native R1/R3 arithmetic and the whole MTP schedule need a separate audit before
production promotion. These initial results are superseded by the fixes below.
Both benchmark processes exit cleanly; Qwen remains
intentionally stopped, with its production files and profile preserved.

### GLM4.10 parity and K8 optimization, November2

The initial token mismatch was reproduced in native single-GPU forwards,
without the network or distributed commit protocol. Serial R1 projections used
different arithmetic from unfused R3 verification. Enabling `EXL3_VERIFY_FUSE=1`
fixes that part. Its paired gate/up launch also picks its reduction partition
using both matrices' total output width; GLM's independent R1 launches pick a
different partition. `EXL3_VERIFY_FUSE_PAIR=0` bypasses that paired launch while
retaining the fused single-matrix verification kernels. This restores all288
measured tokens, plus16 warmup tokens, against the original plain-R1 baseline.
The switch defaults to1, so prior Qwen behavior is preserved.

The main speed loss was GLM's K8 attention matrices missing the fast dense-K8
shape whitelist. Decode reconstructed FP16 weights on every call. Seven actual
GLM KDA/MLA/indexer shapes now qualify under the existing explicit
`EXL3_DENSE_K8=1` switch. Model files and weight precision are unchanged.
Real-weight checks cover112 seed/row/output-dtype cases, scratch/counter bounds,
unchanged inputs, repeatability and exact serial/batched outputs. Error against
independent FP64 Hadamard/matmul stays below0.000306 relative L2. Projection
speedups are3.1–6.3×. This is numerical kernel validation, not a formal model
quality benchmark.

Full two-host results with K8 enabled, VF1/PAIR0 and shared weight decoding:

| Fixed draft depth | Aggregate generation tok/s | Proposal matches |
| --- | ---: | ---: |
| 1 | 13.74 | 116/169 |
| 2 | 13.98 | 145/280 |
| 3 | 13.02 | 154/396 |

All864 measured MTP tokens match optimized serial-R1 reference outputs.
Each depth uses three96-token continuations after its own warmup. Depth2 is
the current preset, about2× the initial7.0 tok/s. Its target-verification round
fell from257 to132ms. The serial-reference timing includes unused draft-cache
catch-up and must not be reported as pure target-only throughput. K8 changes
rounding relative to the old FP16-GEMM path; parity is checked between R1 and
MTP under the same optimized kernels.

Per-block graphs pass exact-output checks and actually replay, but add no speed
on either engine path tested. They remain off. The tested parameters are saved
in the experimental GLM profile. This is still
a2048-context greedy benchmark configuration; production serving, sampled MTP
and full-length context qualification remain separate work.
The supplementary native first-eight-block audit also passes after sparse
attention activates beyond2,048 tokens, including a four-row window at2,051
with173 module tensors exact. This is not full-model long-context qualification.

### GLM serving pilot: two262k sessions

`tools.multinode.serve_glm` uses GLM's native template, reasoning and tool parser
with fixed-depth2 MTP and the qualified K8 settings. Managed user units
`kyojin-glm-cluster.service` on llama and `kyojin-glm-cluster-worker.service` on
llama2 reserve two disjoint262144-token slots (524288 total cache tokens).
Target Q8 MLA backing totals3.52GB/2.94GB on the two stages; KDA states,
MTP cache, weights and runtime buffers are additional. Model memory allocates
at startup rather than growing by the advertised token limit.

The public existing port8731 advertises `glm-5.3-flash-yamz`; OpenClaw's matching
model context is262144 and its existing GLM80% inline compaction remains.
Only that model's context field was changed through validated CLI patching.
Qwen configuration/files are preserved and Qwen is intentionally stopped.
Neither cluster unit is enabled for boot; a fresh pair is required after a
worker/transport failure. The service guard enforces the GPU lock and4GiB
MemAvailable minimum. An authenticated ping every15s prevents the benchmark
worker's60s idle timeout, serializes with model transactions, and marks the
engine unavailable if the worker fails.

This pilot supports greedy text MTP. Sampled decoding, vision and persistent
prefix reuse remain unavailable. Every request prefills its prompt again.
Concurrent4.1k-token requests pass with both full-size caches allocated;
this is not an end-to-end262k prompt stress test.

OpenClaw provider consolidation now uses configuration key`llamaCluster`
(canonical ID`llamacluster`). Its two choices retain Qwen6.05 and add the
GLM4.10 two-node setup. Model rules and saved selections use the new provider
references; old`localqwen`/`localglm` providers have been removed.
