# Deployed cluster baseline: November 5, 2026

This branch preserves the deployed two-node DeepSeek text-inference work before
an upstream update. Its source base is
`ad1a0d84cff8d271c590bca41703059a9efa8a15`. It includes the distributed runtime,
kernel and cache changes, regression tests, and a reviewed configuration example.
The publication did not update or restart the live services or run GPU tests.

## Active configuration

| Item | Audited deployment |
| --- | --- |
| Model | `turboderp/DeepSeek-V4.1-Flash-1017-exl3`, 2.68 bpw |
| Model revision | `608c237e95188739612e435a935ffa614b9c40e9` |
| Coordinator | Embedding and target layers `[0, 20)`; module span `[0, 22)` |
| Worker | Target layers `[20, 40)`, output head and MTP; module span `[22, 44)` |
| Context | 262,144 tokens per session, two sessions, 524,288 total active cache tokens |
| Drafting | Requested MTP depth 3; sampled target verification of deterministic drafts |
| Sampling defaults | Temperature 1.0, top-p 0.95; repetition guard enabled |
| Prefill | Chunk 256, maximum request rows 1,024, configured pipeline window 16 |
| Prefix reuse | Enabled; configured 2 GiB budget per node, 16 entries, 4,096-token grid |
| Engram | Original FP8 pack, disk-backed direct row reads |
| Physical-memory guard | 2 GiB minimum available RAM on each host |
| Host allocator override | `MALLOC_MMAP_THRESHOLD_=1048576` on the coordinator only |

The [coordinator profile example](../examples/multinode/ds41-268-coordinator.json)
records the selected environment flags, including worker-side sampling and the
guarded CPU top-p fast path. The audited worker profile leaves the malloc
override unset. Prefix capacity is additional host state; it is not another
advertised active-session context pool. A configured full cache allocation does
not establish successful generation at maximum context or simultaneous maximum
occupancy.

The HTTP entry point is `python -m tools.multinode.serve_ds41`. It supports
native DeepSeek chat formatting and tool parsing, JSON/SSE replies, seeded
sampling, cancellation, and reuse of released slots. Retained Qwen and GLM
adapters are separate entry points. A model-switching gateway is **not
implemented**: an API model name does not load a different model, coordinate a
service transition, or restore a previous runtime. Standalone Qwen runs from a
separate preserved installation and is not replicated on the worker by this
snapshot.

## Source and native-runtime audit

The read-only November 5 audit compared the two live hosts:

- All **837 audited core files** were byte-identical between hosts.
- All **35 `tools/multinode` Python files** were byte-identical between hosts.
- Both hosts used native-extension SHA-256
  `9166e16baaf2c6be094bdd66b458993f45c3a2c7cd88b8519961da5195c7759a`.
- Common installed versions included Torch
  `2.14.0+rocm10.1.0a20260822`, the ROCm SDK
  `10.1.0a20260822`, Triton `3.8.0+git675c5987.rocm10.1.0a20260822`,
  tokenizers `0.23.3`, NumPy `2.5.3`, and safetensors `0.8.0`.

This branch normalizes source line endings for review. Its source text matches
the captured deployed source after line-ending normalization; its raw file
hashes and distributed handshake fingerprint must not be described as identical
to the live snapshot. `source_fingerprint()` hashes the bytes of core source and
every Python file under `tools/multinode`, including diagnostic and guard
scripts. Deploy the same complete revision to **both** nodes, validate their
fingerprints, and independently qualify any rebuilt native extension. A matching
Git base alone is insufficient when either tree has local changes.

The audit was narrower than whole-machine equivalence. The worker retained more
utility and test files. Several helper/configuration files differed only in line
endings, and one DS41 CPU test had additional lines on the worker. Four GLM
helper paths on the coordinator were malformed symlink placeholders
(`serve_metrics.py`, `image_cache.py`, `serve_ping.py`, `serve_reqlog.py`). These
findings do not contradict core/cluster parity, and they do not establish that
every optional GLM helper is deployable there. Virtual-environment file contents
and JIT-cache binaries were not comprehensively hashed. Native binaries, model
weights, authentication tokens and machine-specific diagnostics are not supplied
by this source snapshot.

## Coordinator allocator finding and persisted fix

A reproduced long-chat workload used 100 user/assistant pairs, a 26,188-token
prompt, 16 output tokens, seed 42, and `cache_prompt=false`. Across two baseline
requests, post-request RSS rose from 7,011 to 10,118 MiB while live heap stayed
near 1,244-1,262 MiB and prefix state stayed at 217.5 MiB. Freed glibc arena space
grew from 4,348 to 7,425 MiB. This reproduction implicated allocator retention
rather than accumulating live prefix tensors.

With `MALLOC_MMAP_THRESHOLD_=1048576`, the two post-request RSS measurements were
2,232 and 2,186 MiB; free arena space was 30 and 31 MiB. The candidate request
times were 102.2 and 93.8 seconds, versus 92.5 seconds for each baseline request.
Prior warmup histories differed, so these are bounded observations, not a clean
performance comparison. A subsequent 256-token decode took 23.31 seconds versus
23.3/23.0 seconds in the baseline. A prefix check reused 7,840 of 7,841 prompt
tokens, with 14.1-second cold and 1.7-second hit requests and stable RSS near
2,203-2,204 MiB. Final health was OK with the same context/session geometry and
approximately 25 GiB physical RAM available.

The only persisted profile change from that investigation is the coordinator's
malloc threshold. Diagnostic `PYTHONPATH` and `KYOJIN_MEMORY_DIAG` settings were
removed from the saved profile. The process inspected at publication time still
retained its original passive 15-second diagnostic sampler until its next
restart; that transient instrumentation is not a permanent configuration
requirement. The worker profile was not given this allocator override.

The historical process that reached roughly 23 GiB RSS had no allocator dump,
and the exact native allocation callsite was not traced. No eight-hour soak was
completed. The fix addresses the reproduced retention pattern; it is not proof
that all long-session memory growth is resolved.

## Reuse and validation limits

`ds41_service_guard.py` and `glm_service_guard.py` are the preserved deployment
wrappers. Their hostnames, filesystem paths, interfaces, service names and
credential locations are specific to the original installation. Review and
adapt those settings before use, keep tokens outside Git and process arguments,
and apply identical source changes on both nodes. The example profile is a
configuration record, not an unattended installer or a portable binary bundle.

The [historical qualification record](multinode.md) describes earlier dense,
Qwen and GLM comparisons. Later DeepSeek work added prefill/cache optimizations,
sampled MTP, worker-side sampling, repetition handling and the allocator fix
above. Their bounded parity, HTTP, prefix and cancellation checks do not imply
universal model-quality equivalence, a maximum-context stress-test pass, or a
throughput result on other hardware. Distributed vision and automatic model
switching are not qualified by this snapshot. Publishing it performs no new GPU
validation and makes no new speed or quality claim.

## Publication checks

The isolated publication checkout passed all 153 unittest executions across 29
CPU suites, with GPU devices hidden and the existing native extension reused.
Each suite ran in a separate process to avoid fixture interference. The checks
covered transport, session lifecycle, speculative verification, HTTP handlers,
sampling, prefix storage, memory accounting and prefill options. All 102 changed
or added Python files parsed successfully. All 872 captured core and cluster
source files matched after line-ending normalization, and the staged diff
passed Git's whitespace check. These checks did not restart either service.
