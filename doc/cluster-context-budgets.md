# DeepSeek cluster context and retained-cache budgets

The DeepSeek cluster has fixed active session slots and a separate host-memory
prefix store. `--context` limits the complete token window of each active slot;
`--sessions` selects concurrency. `--cache-tokens` optionally bounds the total
of active slot capacity plus logical retained prefix tokens. It must be at least
`context * sessions`. Additional tokens require prefix caching to be enabled.

For example, `--context 131072 --sessions 2 --cache-tokens 393216` allocates two
131072-token active slots and permits at most 131072 retained prefix tokens.
It does not allocate a single shared 393216-token GPU KV arena. Retained
checkpoints continue to obey the existing per-node physical RAM budget and
entry limit; the tighter limit controls eviction. Overlapping snapshots count
their full logical lengths conservatively, even when host pages are shared.
Token capacity is therefore an upper bound, not a promise that every token of
retained capacity will be reachable under the physical byte budget.

`/health` and `/v1/models` expose `active_cache_tokens`, `retained_cache_tokens`,
and their sum as `cache_tokens`. `prefix_stats.retained_tokens` reports current
retained occupancy, separately from cumulative cache hits/reused tokens.
The optional token cap defaults to unset for backwards compatibility; omitted
`--cache-tokens` retains the earlier physical-budget-only prefix policy.

The managed DeepSeek guard takes its default session context and total token
budget from its profile. An explicit guard `--context` remains a qualification
override with no additional retained-token allocation. The two managed roles
must use the same session geometry and matching source fingerprints. Existing
fusion, MTP, sampling and physical-memory watchdog controls are independent.

CPU regression covers bounded prefix eviction across distinct prompts, exact
repeat restoration, zero-token retention, the existing physical-memory limit,
two sessions and HTTP capacity metadata. Full-model qualification should verify
prefix reuse, cancellation/session release and physical headroom after loading.
