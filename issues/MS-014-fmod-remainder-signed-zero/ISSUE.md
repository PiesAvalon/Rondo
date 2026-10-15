# `fmod` and `remainder` return `+0.0` for every zero result, dropping the sign required by C99 and Python

**Component:** MindSpore 2.9.0 CPU, `mindspore.ops.fmod` / `mindspore.ops.remainder` and the
torch-compatible `mindspore.mint.fmod` / `mindspore.mint.remainder`, PYNATIVE and GRAPH (`ms.jit`),
float32 and float64
**Severity:** wrong bit-level result (a signed zero becomes `+0.0`); silent, and it changes
`1/r`, `signbit`, `copysign` and comparisons against `-0.0` for a zero remainder.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** a `mindspore.mint` semantics differential against NumPy
(`artifact/hunt-v2/tools_ms_mint_probe.py` + the semantics table), minimised to single pairs.

---

## Summary

A zero result of a remainder operation still has a sign, and both conventions are unambiguous:

* `fmod(x, y)` - C99, `math.fmod`, `np.fmod` - takes the sign of the **dividend**;
* `remainder(x, y)` / Python `%` / `np.remainder` - takes the sign of the **divisor**.

MindSpore returns `+0.0` in every cell whose exact result is a negative zero:

| call | MindSpore (`ops` and `mint`) | NumPy / C99 / Python |
|---|---|---|
| `fmod(-0.0, 1.0)` | `+0.0` | `-0.0` |
| `fmod(-1.0, 1.0)` | `+0.0` | `-0.0` |
| `fmod(-2.0, 2.0)` | `+0.0` | `-0.0` |
| `fmod(-0.0, -1.0)` | `+0.0` | `-0.0` |
| `remainder(0.0, -1.0)` | `+0.0` | `-0.0` |
| `remainder(1.0, -1.0)` | `+0.0` | `-0.0` |
| `remainder(2.0, -2.0)` | `+0.0` | `-0.0` |
| `fmod(0.0, 1.0)`, `remainder(-0.0, 1.0)` | `+0.0` | `+0.0` (controls) |

Non-zero results are correct, which is why this is easy to miss: `+0.0 == -0.0` is true, so only
`signbit`, `1/r` (`+inf` vs `-inf`) or `copysign` can observe it.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(166 checks per replay.)

```python
import numpy as np, mindspore as ms
from mindspore import ops
import mindspore.mint as mint
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

x, y = ms.Tensor(np.float64(-1.0)), ms.Tensor(np.float64(1.0))
np.signbit(np.asarray(ops.fmod(x, y).asnumpy()))       # False  <-- should be True (-0.0)
np.signbit(np.fmod(np.float64(-1.0), np.float64(1.0)))  # True
1.0 / float(ops.fmod(x, y).asnumpy())                  # +inf   <-- should be -inf
```

What the reproducer pins:

| dimension | result |
|---|---|
| ops | `fmod` **and** `remainder` both drop the sign |
| layers | `mindspore.ops.*` and `mindspore.mint.*` behave identically (a shared kernel) |
| dtypes | float32 and float64 |
| lanes | `ms.jit` (GRAPH) returns `+0.0` as well |
| cells | 8 zero-result pairs per operation, each pinned on both sides plus the NumPy reference and the convention it follows |
| controls | non-zero results (`5 % 3`, `-5 % 3`, `1e8 % 3`, ...) match NumPy to 1e-12, and the pairs whose NumPy result is `+0.0` are `+0.0` in MindSpore too |

## Root cause (pointer)

The zero-result path of the CPU `Fmod`/`Remainder` kernels constructs the result from the magnitude and
the integer part without restoring the sign: C99 requires `fmod` to return a zero with the sign of `x`,
and Python semantics require `remainder` to return a zero with the sign of `y`.  Computing the sign
explicitly (`copysign(0.0, x)` for `fmod`, `copysign(0.0, y)` for `remainder`) when the remainder is
zero fixes both layers, since they share the kernel.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `fmod signed zero`, `remainder negative zero`,
  `fmod negative zero`, `signed zero mod` and `fmod sign` returns **nothing**; a web search surfaces
  only implementation tasks (PyBoost/aclnn `fmod` adaptation), not a bug report.
* Our own MS-001 draft is a *different* defect in the same operator: GRAPH-mode `fmod` returning wrong
  **values** (`fmod(1e12, 3.0) = 4096`) from a reduced-precision quotient, and it uses `remainder` as a
  value control - which this finding shows was only half true: `remainder`'s values are right, its zero
  *signs* are not.
