# `ops.fmod` returns wrong values in GRAPH mode (`ms.jit`) while PYNATIVE mode is correct

**Component:** MindSpore 2.9.0, `mindspore.ops.fmod` — GRAPH mode lowering (AKG kernel)
**Severity:** silent wrong result (wrong sign, wrong magnitude, and results outside the
documented `|fmod(a,b)| < |b|` range)
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** Rondo edge-value differential hunt (`artifact/hunt-v2`, dtype lanes)

---

## Summary

`ops.fmod(a, b)` must satisfy `fmod(a, b) = a - trunc(a / b) * b`, `|fmod(a, b)| < |b|` and
`sign(fmod(a, b)) == sign(a)`. In PYNATIVE mode MindSpore does exactly that (matching NumPy); under
`ms.jit` (GRAPH mode) the same call returns wrong values, including results **larger than the
divisor** and with the **opposite sign**.

```python
import mindspore as ms
from mindspore import ops

x = ms.Tensor([1e12], ms.float64)
y = ms.Tensor([3.0], ms.float64)
ops.fmod(x, y)          # [1.0]     correct (1e12 = 3*333333333333 + 1)
ms.jit(ops.fmod)(x, y)  # [4096.0]  wrong: |4096| > 3, and 1e12 - 4096 is not a multiple of 3
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| `fmod(a, b)` (float64) | PYNATIVE | GRAPH (`ms.jit`) | NumPy | comment |
|---|---|---|---|---|
| `fmod(1e8, 3.0)` | `1` | **`-2`** | `1` | wrong value, wrong sign |
| `fmod(1e9, 3.0)` | `1` | **`-32`** | `1` | wrong value |
| `fmod(1e12, 3.0)` | `1` | **`4096`** | `1` | **result larger than the divisor** |
| `fmod(1e8, 1e-8)` | `0` | **`-2.726`** | `7.9e-09` | result 8 orders of magnitude too large |
| `fmod(1e10, 1e-5)` | `9.5e-06` | **`130.09`** | `9.2e-06` | out of range |
| `fmod(-1e8, 3.0)` | `-1` | **`2`** | `-1` | sign flipped |
| `fmod(1e8, -3.0)` | `1` | **`-2`** | `1` | sign flipped |
| control `fmod(7.0, 3.0)` | `1` | `1` | `1` | agrees |
| control `fmod(12345.678, 1.0)` | `0.678` | `0.678` | `0.678` | agrees |
| control `fmod(1e6, 3.0)` | `1` | `1` | `1` | agrees |

The same inputs through the neighbouring operators are **correct in GRAPH mode**, so this is
specific to `fmod`:

| op, `a = 1e9`, `b = 3.0`, float64 | PYNATIVE | GRAPH |
|---|---|---|
| `ops.fmod` | `1` | **`-32`** |
| `ops.remainder` | `1` | `1` |
| `ops.floor_divide` | `3.3333333e+08` | `3.3333333e+08` |
| `ops.truncate_div` | `3.3333333e+08` | `3.3333333e+08` |
| `ops.div` | `3.3333333e+08` | `3.3333333e+08` |

The error grows with `|a/b|`: it is absent for small quotients (`1e6/3`, `7/3`,
`12345.678/1.0`) and appears once the quotient exceeds what a 24-bit mantissa can represent
(`1e8/3` needs 27 bits, `1e12/3` needs 40 bits), which points at the GRAPH kernel evaluating the
quotient in a reduced precision (a float32 intermediate / reciprocal-multiply explains the
`-32` and `-2` cases exactly: `float32(1e9) / float32(3) = 333333344`, `* 3 = 1000000032`,
`1e9 - 1000000032 = -32`).

## Root cause (pointer)

`fmod` in GRAPH mode is lowered to a fused kernel whose `a / b` (or `a * (1/b)`) step is evaluated
in reduced precision, so `trunc(a/b)` is off by large integers and the final subtraction lands far
outside `[0, |b|)` (or `(-|b|, 0]`).  PYNATIVE mode uses the reference implementation and is
correct.  The fix belongs in the GRAPH/AKG `Fmod` kernel: compute the quotient in the input dtype
(or use the platform `fmod` after the type is preserved).

## Impact

`fmod`/`%` on large magnitudes silently returns a wrong number in graph mode, and the result can
exceed the divisor — so downstream code that relies on `fmod(a, b) < b` (modular indexing, angle
wrapping, hashing, periodic padding, sequence-position computation) can index out of range or
produce a completely wrong value.  Model code compiled with `ms.jit`/GRAPH mode is affected while
the same code in PYNATIVE is not, so the bug appears only after enabling graph compilation.

## Novelty assessment

* Tracker searches in `mindspore-ai/mindspore` for `fmod wrong result`, `fmod graph mode` and
  `fmod precision float32` return no matching report.
* The draft is deliberately scoped to the GRAPH-vs-PYNATIVE divergence with an out-of-range
  result, which is the unambiguous part (a wrong sign and `|result| > |b|` cannot be defended as
  a precision difference).

## Files

* `repro.py` — self-checking reproducer (per-case GRAPH/PYNATIVE/NumPy table, out-of-range and
  sign checks, control set, neighbouring-op controls)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
