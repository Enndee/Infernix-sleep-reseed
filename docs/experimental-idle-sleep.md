# Experimental Qwen4Exp idle VRAM sleep (native Windows)

This opt-in experiment preserves the running process, model pointers, CUDA graphs
and conversation state while releasing physical GPU backing at idle execution
boundaries. It applies to Qwen4Exp only and does not change default behavior,
command-line flags, HTTP routes or any production launcher.

The implementation and measurements are based on `windows-2026.10.18`
(`a8a408edc5dbf3a81a76651285db24d3124df226`), plus the prerequisite expert-cache
sleep/reseed commit `ee06da447da47b36c6eedb81083da92d6685ceb2`. Upstream `main`
had advanced by 108 commits to `5bc33120595f8cf437c983c420d568c7af1c1380` when
this work was prepared for review on 25 October 2026. These results do **not**
qualify a forward port to that newer execution path. Linux is also unverified.

## Controls

Set environment variables before launching the native server:

```powershell
$env:INFERNIX_SLEEP_AFTER_SECONDS = '15'
$env:INFERNIX_SLEEP_KEEP_FRAMES = '0'
$env:INFERNIX_SLEEP_SCRATCH = '1'
$env:INFERNIX_SLEEP_WEIGHTS = '1'
```

- `INFERNIX_SLEEP_AFTER_SECONDS`: `0`/unset disables automatic sleep. Requires
  the VRAM monitor and elastic VMM expert pools.
- `INFERNIX_SLEEP_KEEP_FRAMES`: expert-cache floor; default zero.
- `INFERNIX_SLEEP_SCRATCH=1`: opt in to VMM-backed static workspace and expert miss staging.
  Their contents are disposable; every virtual address stays reserved. Free KV page
  tails may shrink below the normal startup floor, but live/prefix-owned pages are
  never removed to meet that floor. The normal floor is restored before binding.
  Wide workspaces/streaming rings already borrow expert frames, so there is no
  second persistent allocation to free for them.
- `INFERNIX_SLEEP_WEIGHTS=1`: use VMM backing for materialized resident device weights and
  capture one exact pinned-host mirror before arming sleep. Requires additional RAM
  equal to the resident weight arena, allocated once. Does not read model files on
  wake. Host expert banks and pinned vision weights are unchanged.

Each backing option is independent; keep both new settings unset/zero for the previous
expert-only behavior. Both new settings use the *same* sleep threshold, not a second
delayed deep-sleep tier. No memory is unmapped while any lane or borrowed frame
lease is live. The execution boundary explicitly synchronizes all CUDA streams.

Scratch backing is enabled only with the monitor and a nonzero idle threshold.
The weight option is consumed earlier, during model materialization: setting it
alone still selects VMM backing and reserves the later pinned-memory budget, even
if automatic sleep is disabled. The mirror is captured when sleep is armed. To
restore the default allocation path, unset **both** backing options and the idle
threshold before restarting the server. Startup does not silently fall back when
explicitly requested VMM backing is unavailable. Sleepable materialization cannot
share a vision eviction pool; host-resident/offloaded vision weights are unchanged.

## Ownership and transition ordering

`SleepableBuffer` wraps the existing `VmmArena`. Its virtual reservation lives for
the buffer's lifetime; only physical chunks are released. It has two uses:

- **Scratch:** no mirror; contents are disposable and are rewritten by consumers.
- **Resident weights:** one exact pinned-host copy captured after initialization;
  wake copies these bytes back without rereading or repacking the model artifact.

The artifact owns the resident-weight buffer. The Program owns scratch/staging
buffers, declared to outlive their non-owning workspace and Forward views. The
additional weight mirror is included in startup pinned-RAM budgeting before the
expert and host tiers consume their share. Other model families retain their
existing materialization path.

At an idle execution boundary, sleep:

1. Calls `cudaDeviceSynchronize()` **before the first unmap**, including KV tails.
   The compute-stream-only `DeviceContext::synchronize()` is insufficient here.
2. Trims only unreferenced KV tails below the normal startup floor when scratch
   sleep is enabled; it does not evict retained prefixes to meet a numeric target.
3. Saves the expert ranking/counts, releases frames down to the keep floor, and
   defers the transition if outstanding leases prevent the requested release.
4. Publishes the asleep state before unmapping scratch, staging and weights, so
   even a partially failed transition must pass the wake barrier before access.

On wake, resident weights, scratch, staging and the normal KV floor are restored
**before** optional expert growth/reseeding. `SleepableBuffer` rolls back its own
new mappings on allocation failure and remains asleep. Essential allocation
failure prevents inference and raises a recoverable execution error; the virtual
addresses and mirror remain available for a later retry. Allocation-failure
injection has not been qualified. A partial/zero expert-cache restore is distinct
from essential-buffer failure and retains the host/staging fallback.

`SleepableBuffer` is not a synchronizing allocator: callers must serialize
transitions and quiesce every stream using the storage before `sleep()`. Capturing
an immutable mirror is one-time; later `capture()` calls do not refresh it.

Essential weights, scratch and normal KV backing restore **before** expert-cache
growth/reseeding and before source selection/admission, not just request binding
(which can already execute vision). Admission estimates growable KV from currently
resident expert frames: quoting while they sleep can falsely reject a retained
snapshot and force a full recomputation. A larger pre-existing KV pool can hide
this ordering bug; fresh-server retained-context repeats are necessary.
The normal round preambles remain a fallback wake barrier. If essential backing
cannot be mapped, inference does not access the unmapped pointers. Zero restored
expert frames remains supported by the existing host/staging path.

## Evidence and diagnostics

`sleep memory asleep` records separately released scratch and resident-weight MiB.
`sleep memory awake` records essential restore duration; `expert cache awake`
includes that work plus expert remapping and reseeding. KV and recurrent/prefix
state remain owned by their original pools. Desktop/system usage is not the same
as the server's process-owned dedicated VRAM.

Focused test target: `infernix_sleepable_buffer_test`. It checks five cycles of
physical release, identical pointers, exact mirrored contents including a partial
last chunk, scratch writes, and replay of the *same* pre-sleep CUDA graph executable.
Existing arena/decode-graph/KV tests should run alongside it.

In a configured, test-enabled Windows build tree:

```powershell
cmake --build build-windows --config Release --parallel --target infernix-serve infernix_sleepable_buffer_test infernix_arena_test infernix_decode_graph_test infernix_kv_cache_test
ctest --test-dir build-windows -C Release --output-on-failure -R '^infernix_(sleepable_buffer|arena|decode_graph|kv_cache)_test$'
```

`infernix_sleepable_buffer_test` skips with code 77 if CUDA/VMM is unavailable;
the reported GPU validation actually ran and passed, rather than skipping.

The [sanitized per-sample evidence](measurements/qwen4-idle-sleep-2026-10-25.json)
contains all four campaigns, complete result ranges, output hashes, retained-prefix
checks, selected engine logs and source/executable hashes. Local full artifacts
remain outside Git; no workstation paths, process IDs or launch credentials are
needed to review the committed evidence. The short-prompt campaigns use the
same 42-token prompt, 96 output tokens, temperature zero, thinking disabled, 15 s
sleep, four post-sleep requests and four immediate warm follow-ups. Stream timing
starts before HTTP dispatch and ends TTFT at the first nonempty content delta.
Each post-sleep request requires matching sleep/wake events from the same port-8011
server log. GPU/process counters and decoded text are retained per sample.
Because wake now happens during source admission, native request logs can charge
it to `queue` time instead of their TTFT. Compare the client stream's end-to-end
TTFT (HTTP dispatch through first content), not native TTFT with wake omitted.

Do not claim general 262K-context, vision, concurrency, or long-conversation
qualification from a short-prompt latency benchmark. Preserve incomplete runs and
report interference rather than selecting only favorable samples.

## Measured results - 25 October 2026

All rows have four post-sleep samples and four immediate warm follow-ups. These
are sequential workstation runs, not randomized trials; baseline predates the
new implementation. The latest build also fixes both staging calibration consumers.

| Configuration | Median post-sleep TTFT | Median warm TTFT | Median idle dedicated VRAM | Engine decode tok/s, post-sleep / warm |
|---|---:|---:|---:|---:|
| Expert-only baseline | 0.789 s | 0.263 s | 8209.3 MiB | 68.55 / 74.50 |
| Scratch-only (pre-final build) | 0.800 s | 0.241 s | 6727.3 MiB | 68.40 / 81.50 |
| Scratch + weights (pre-final build) | 0.982 s | 0.235 s | 857.8 MiB | 76.65 / 83.75 |
| Scratch + weights (final build) | 1.045 s | 0.266 s | 841.8 MiB | 77.05 / 80.85 |

The final build's post-sleep TTFT range was 0.965-1.131 s; essential restoration
median 0.376 s, range 0.333-0.427 s. Versus the earlier expert-only baseline this
is +0.256 s median TTFT for ~89.7% less idle dedicated memory. These small runs
do not isolate decode-throughput effects; do not interpret differences as a speedup.
All 32 measured short-prompt completions (four configurations, eight per row)
exactly matched baseline text and produced 96 output tokens.

The engine separately logged 928 MiB scratch, 5888 MiB resident weights and
10880 MiB expert frames released. The exact pinned-host weight mirror costs about
5.75 GiB additional RAM. Stable virtual reservations remain, and the GPU context,
recurrent state and prefix-owned KV are intentionally not discarded.

Retained-context validation used an explicit system-message cache breakpoint:
5162 prompt tokens, 5120 reused tokens (99.2%), exact `SILVER-OTTER-731` output
before and after **four actual sleep/wake cycles**. Post-sleep TTFTs were
1.002 / 1.098 / 1.059 / 0.961 s. The final asleep dedicated counter with that
retained context was **1039.8 MiB**, higher than the short-prompt floor by design.
System-wide GPU usage at that point was 5561 MiB, not 1039.8 MiB.

Final Release rebuild and the four focused CTest tests passed (1.50 s total).
The buffer test compares all mirrored bytes, verifies capture is one-time and
replays the same graph for five cycles. Six unchanged supervisor tests passed.
Final job and audit markers were `0`; `/v1/models` answered HTTP 200 without waking
inference. The isolated server was left asleep; production configuration was
unchanged. The external supervisor's six mocked tests are not native-engine
coverage and are not part of this repository's CTest results.

**Review follow-up completed:** final results above include all-stream
synchronization before the first free-KV-tail unmap and the stronger buffer test.
The earlier passing admission-fix run (1.091 s median post-sleep TTFT) is preserved
locally, separately from the final campaign; it is not substituted for final evidence.
The stronger test initially forgot to settle its deliberate default-stream
overwrite before unmap; test synchronization was fixed without relaxing byte
assertions. The failed test run is preserved locally.

The failed fresh-server cache-admission run is preserved locally and summarized
in the committed evidence: a warm 5120-token hit became zero reused tokens after
sleep, with 2.874 s client TTFT. Waking only at binding was too late because source
admission had already rejected the retained snapshot. Earlier long-prompt probes
that never reused a prefix, even when warm, are not counted as retention tests.

Still unqualified: near-262K prompts, vision inference, concurrent requests,
allocation-failure injection, full-suite CTest and long-duration soak.
No HTTP sleep/wake control plane or host-expert-memory release is implemented.