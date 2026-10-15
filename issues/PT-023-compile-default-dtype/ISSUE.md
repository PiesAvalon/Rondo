# `torch.compile` ignores `torch.set_default_dtype`

**Component:** PyTorch 2.12.1 (CPU, Inductor + Dynamo), `torch.compile(dynamic=False)`, eager vs
compiled, single process and across processes with a shared `TORCHINDUCTOR_CACHE_DIR`
**Severity:** silent wrong dtype **and value** — a compiled function that builds a tensor from the
default dtype keeps using `float32` even when the default is `float64`, and changing the default
after compilation does not invalidate the compiled callable
**Status:** present in torch 2.12.1+cu130 on CPU; the same functions run eagerly (or compiled with
explicit dtypes) produce the correct `float64` results
**Found with:** the wave-178 compile-cache probe (Rondo's Persistent/Reference oracles: a shared
inductor cache and an in-process retained callable), minimised to the witnesses below; certified
reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

A compiled function body is traced once, and the tensors it builds from the **default** dtype are
baked in as `float32` regardless of `torch.set_default_dtype`:

```python
import torch
torch.set_default_dtype(torch.float64)

torch.compile(lambda: torch.randn(4))().dtype          # torch.float32   (eager: torch.float64)

x = torch.ones(4, dtype=torch.float32)
torch.compile(lambda x: x * torch.full((4,), 0.1))(x)
# tensor([0.1000, ...], dtype=torch.float32)   with the float32 constant 0.10000000149011612
# eager:                dtype=torch.float64, value 0.1
```

The wrong dtype also changes the **value** (`0.10000000149011612` instead of `0.1`), so this is not
a cosmetic metadata issue.  The compiled callable is additionally not invalidated when the default
dtype changes:

```python
def mul(x):
    return x * torch.full((4,), 0.1)

torch.set_default_dtype(torch.float32)
fc = torch.compile(mul, dynamic=False)
fc(x)                                   # float32, correct for the current default

torch.set_default_dtype(torch.float64)
fc(x)                                   # still float32 with the float32 constant
mul(x)                                  # eager: float64, 0.1
```

There is no guard on `torch.get_default_dtype()`, so the stale graph is reused both in-process and -
with a shared `TORCHINDUCTOR_CACHE_DIR` - **across processes** (a second process that sets
`torch.set_default_dtype(torch.float64)` before compiling the same function receives the float32
result when the cache was seeded by a float32 process; with a fresh cache it correctly compiles to
float64).

## Scope and controls

* **Explicitly typed constants are correct**: `x * torch.full((4,), 0.1, dtype=torch.float64)`
  compiles to `float64`, so the defect is specific to the *default*-dtype path.
* **The standalone factories are correct**: `torch.ones`, `torch.zeros`, `torch.full`,
  `torch.arange` and `torch.tensor` compiled under `set_default_dtype(torch.float64)` all return
  `float64`; only `torch.randn` and the *use of a default-dtype constant inside a larger expression*
  flip to `float32`.  The trigger is therefore the constant's promotion against an `float32` input,
  not the factory call itself.
* Under the standard `float32` default every case matches eager, so the bug requires a non-default
  dtype.
* `torch.set_default_dtype(torch.float32)` is restored at the end of the reproducer.

## Impact

Models that rely on `torch.set_default_dtype(torch.float64)` (scientific computing, numerical
differentiation, anything needing double-precision accumulates) silently get float32 precision
inside compiled regions, with values rounded at the float32 quantum.  The failure is silent: the
output is a valid tensor, only with the wrong dtype and lower precision, and it can also be pulled
from a stale on-disk cache in a later process.

## Suggested fix

Either guard the compiled artifact on `torch.get_default_dtype()` (forcing a retrace when it
changes) or lower default-dtype factories/constants using the *current* default dtype at runtime.
A regression test can assert `compiled().dtype == torch.get_default_dtype()` for `torch.randn` and
for `x_f32 * torch.full((n,), 0.1)`.

## Novelty

Searched the PyTorch tracker for `torch.compile set_default_dtype`, `dynamo set_default_dtype guard`,
`torch.compile float64 constant dtype`, `torch.compile type promotion constant` and `inductor
float32 constant float64`.  The nearest item is **#138015** ("torch.compile and
torch.set_default_dtype bfloat16 wrong datatype", closed 2024-10), which is the `bfloat16` +
`torch.randn(1, dtype=torch.bfloat16)` case; on 2.12.1 that repro is **fixed** (`randn` with an
explicit bfloat16 dtype now returns bfloat16), while the `float64` default-dtype path reported here
still fails, and the additional facets - a default-dtype constant promoted against an `float32`
input, the value change, and the missing invalidation across an in-process and an on-disk cache -
are not covered there.  #134030 is the closed context-manager case, and #135758 is the unrelated
denormal-flush issue.

## Environment

torch 2.12.1+cu130 (CPU execution), Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64, RTX 4090 present
but all probes ran on CPU.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"`set_default_dtype` after compilation is unsupported."**  The first trigger is a *fresh*
  compile under `set_default_dtype(torch.float64)`, so no "changed after compilation" scenario is
  required; eager and compiled disagree on the same process state.
* **"This is #138015."**  That issue was about an explicit `dtype=torch.bfloat16` in `torch.randn`
  and is fixed on this build; the float64 default-dtype path, the constant-promotion facet and the
  cache-invalidation facet are distinct and still reproduce here.
* **"Float32 is a documented compile default."**  `torch.compile` documents support for float64
  models, and the same functions compiled with explicit float64 dtypes return float64; the default
  dtype is part of the program's semantics (eager honours it), so the compiled path is inconsistent
  with the eager path.
