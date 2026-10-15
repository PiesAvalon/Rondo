# `[aot_autograd]` the AOTAutograd cache is keyed on the **forward** graph only, so changing a custom op's `register_autograd` rule reuses the previous run's backward — silently wrong gradients

**Component:** `torch._functorch._aot_autograd.autograd_cache` (`enable_autograd_cache = True` by default)
**Severity:** **silent wrong gradients** (the forward output stays correct); cross-process
**Status:** present in `torch 2.12.1+cu130`; disabling `torch._functorch.config.enable_autograd_cache` fixes it
**Found with:** Rondo persistent/cross-process oracle on `torch.library.custom_op` + `register_autograd`

---

## Summary

`AOTAutogradCacheDetails` (`autograd_cache.py:403`) inherits from `FxGraphHashDetails` and hashes
the **forward** graph module, the compile config and the triton kernels attached to ops in that
forward graph.  The **backward** graph is not part of the key: for a
`torch.library.custom_op(...).register_autograd(...)` operator it comes from a *separate*
registration and can change while the forward graph stays byte-identical.

With the persistent `TORCHINDUCTOR_CACHE_DIR` (the default user cache behaves the same), a new
process that registers a different backward rule for the same operator therefore loads the
previous process's cached compiled backward.  The forward result is correct and the gradients are
silently the old ones.

```python
# phase.py — run twice, in two processes, with the same TORCHINDUCTOR_CACHE_DIR
import torch
SCALE = float(sys.argv[1])                      # 5 in process 1, 11 in process 2

@torch.library.custom_op("rondo::cp", mutates_args=())
def op(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0                              # forward is identical in both processes
@op.register_fake
def _(t): return torch.empty_like(t)
def setup_context(ctx, inputs, output): ctx.save_for_backward(inputs[0])
op.register_autograd(lambda ctx, g: g * SCALE, setup_context=setup_context)

t = torch.arange(1.0, 5.0, requires_grad=True)
torch.compile(lambda x: torch.ops.rondo.cp(x))(t).sum().backward()
print(t.grad.tolist())
```

```
process 1 (SCALE=5)                     -> [5.0, 5.0, 5.0, 5.0]     correct
process 2 (SCALE=11, same cache dir)    -> [5.0, 5.0, 5.0, 5.0]     <-- STALE, should be [11.0, ...]
```

The two AOT graphs are *different* (`mul(tangents_1, 5.0)` vs `mul(tangents_1, 11.0)`), so the
cache key is not merely "a hash collision" — the backward simply is not part of the key.

## Reproduction results

`python repro.py` (in this directory) drives both processes itself and prints:

| check | observed | verdict |
|---|---|---|
| process 1: `register_autograd(g*5)` | `grad=[5,5,5,5]` | correct (artifact cached) |
| process 2: `register_autograd(g*11)`, shared cache | `grad=[5,5,5,5]` | **stale** — must be `[11,11,11,11]` |
| control: fresh cache dir, `g*11` | `grad=[11,11,11,11]` | correct |
| control: `enable_autograd_cache = False`, shared cache, `g*9` | `grad=[9,9,9,9]` | correct |
| control: forward changed (`t*3`) between processes | `y=[3,6,9,12]` | forward change **is** detected |

5 checks, 0 failures, deterministic across repeated runs.  The symmetric case (11 first, then 5)
reproduces in the same way, and a third distinct rule (`g*7`) also reuses the first cached
backward — the cache entry is reused indefinitely.

## Mechanism

`cache_key_evidence.txt` records the details:

* `AOTAutogradCacheDetails(FxGraphHashDetails)` — hashes the forward GM; its only addition is
  `get_triton_source_codes_from_gm(gm)`, i.e. triton kernels attached to ops *in the forward
  graph*;
* `TORCH_LOGS=aot_graphs` shows the backward graphs differ (`mul(..., 5.0)` vs `mul(..., 11.0)`)
  while the forward graph is identical;
* `enable_autograd_cache = True` and `fx_graph_cache = True` are both defaults, so this happens
  with a plain `torch.compile` run and the default on-disk cache.

## Suggested fix

Include the backward in the key.  The cache already has the machinery to bypass safely
(`BypassAOTAutogradCache`, used e.g. for custom partitioners/solvers that cannot be hashed), so
the minimal fix is either

* add a hash of the traced backward graph (the joint graph's `bw_module` code, which is available
  before the cache lookup completes the AOT pass), or
* bypass the AOTAutograd cache when the forward graph contains a custom op that has a registered
  autograd rule that cannot be hashed (mirroring `EXCLUDED_KWARGS`/`BypassAOTAutogradCache`).

A regression test is the two-process sequence above with a shared temporary
`TORCHINDUCTOR_CACHE_DIR`.

## Related upstream work (novelty assessment)

* pytorch#184405 / PR #184500 are the same *family* — an Inductor cache key that misses an input
  (the default dtype) — but that is the Inductor `FxGraphCache`, not the AOTAutograd cache, and
  the missing input is different.  The `FINDINGS-REPORT` negative-result table already excludes
  the default-dtype case as known.
* PR #194452 (`[precompile] Stop requiring AOTAutogradCache to key a graph before recording it`,
  closed) touches AOTAutogradCache keying, but changes *when* a graph is required to be keyable,
  not which parts of the AOT result the key covers.
* Tracker searches for `AOTAutogradCache backward key`, `enable_autograd_cache wrong gradients`,
  `autograd cache stale backward custom op`, `aot autograd cache key forward graph only` and
  `torch.compile cached backward wrong` return no matching report.

## Impact

Changing a custom operator's backward rule is a normal thing to do — switching between an exact
and an approximate backward, between fp32/fp64 rules, between a library's implementations, or
adding an instrumentation wrapper.  Because the cache is on disk and shared across processes, the
stale backward survives `torch._dynamo.reset()`, a new interpreter and a fresh
`torch.compile` call; only clearing the cache (or `enable_autograd_cache=False`) fixes it.  The
failure is silent: the forward output is correct, so only gradient-based comparisons reveal it.

## Files

* `repro.py` — self-driving two-process reproducer (5 checks, exit code = number of failures)
* `cache_key_evidence.txt` — AOT graph diff, key class inspection, observed matrix, environment
* `collect_env.txt` — environment and relevant configuration defaults
