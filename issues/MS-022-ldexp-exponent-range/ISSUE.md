# `ops.ldexp` evaluates `2**n` in float32, so float64 results beyond the float32 exponent range collapse to `inf`, `0` or `nan`

**Component:** MindSpore 2.9.0, CPU, PYNATIVE and GRAPH (`ms.jit`), `ops.ldexp` on `float64`
**Severity:** silent wrong value — results that are ordinary, representable doubles come back as
`inf` (or `0`, or `nan`), with an exact, reproducible boundary at the float32 exponent range
**Status:** present in MindSpore 2.9.0 (CPU); NumPy, PyTorch, JAX and `math.ldexp` return the finite
float64 values; the float32 lane of `ops.ldexp` is correct
**Found with:** the wave-165 stepping/scaling grid (`artifact/hunt-v2/results/w165_*`), triaged by
Rondo's `check_consensus` oracle with `math.ldexp`/NumPy as arbiter; the certified reproducer is
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

`ops.ldexp(x, n)` is documented as `x * 2**n`.  For float64 inputs it appears to compute the
**scale factor `2**n` at float32 precision**, so the scale overflows to `inf` for `n >= 128` and
underflows to `0` for `n <= -150`, taking any float64 `x` with it:

```python
import numpy as np, mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
def T(x): return ms.Tensor(np.asarray(x), ms.float64)

print(ops.ldexp(T(1.0), ms.Tensor(128, ms.int32)).asnumpy())    # inf      <- 3.402823669209385e+38
print(ops.ldexp(T(1e-300), ms.Tensor(1023, ms.int32)).asnumpy())# inf      <- 8.98846567431158e+07
print(ops.ldexp(T(1.0), ms.Tensor(-150, ms.int32)).asnumpy())   # 0.0      <- 7.006492321624085e-46
print(ops.ldexp(T(1e308), ms.Tensor(-150, ms.int32)).asnumpy()) # 0.0      <- 7.006492321624085e+262
print(ops.ldexp(T(0.0), ms.Tensor(128, ms.int32)).asnumpy())    # nan      <- 0.0
print(ops.ldexp(T(np.inf), ms.Tensor(-150, ms.int32)).asnumpy())# nan      <- inf
```

The boundary is exactly the float32 exponent range, which pins the root cause:

| `n` | MindSpore | NumPy float64 | comment |
|---|---|---|---|
| `127` | `1.7014118346046923e+38` | same | `2**127 <= FLT_MAX` |
| `128` | **`inf`** | `3.402823669209385e+38` | `2**128 > FLT_MAX`, but finite in float64 |
| `-149` | `1.401298464324817e-45` | same | smallest float32 subnormal (`2**-149`) |
| `-150` | **`0.0`** | `7.006492321624085e-46` | `2**-150` underflows float32 |
| `1023` | **`inf`** | `8.98846567431158e+307` | float64 result is finite |

With `x = 1e-300` the result `1e-300 * 2**1023 = 8.99e+07` is not even large - it is `inf` because
only the scale is out of range.  Two further classes follow from `0 * inf` and `inf * 0`:
`ldexp(±0.0, n >= 128)` is `nan` (should be `±0.0`) and `ldexp(±inf, n <= -150)` is `nan` (should be
`±inf`).  The defect affects scalars and vectors, eager and `ms.jit`, while the **float32 lane is
correct** at the same boundaries (`n = 128` genuinely overflows float32 there), and genuine float64
overflow/underflow (`n = 1024`, `n = -1075`) is reported correctly - so the kernel simply uses the
wrong precision for the scale in the float64 lane.

## Suggested fix

Compute the scale in the input's dtype (or, better, implement `ldexp` by direct exponent
manipulation as the C `ldexp` does, which also handles the subnormal range exactly), instead of
`x * cast<float32>(pow(2, n))`.  A regression test should include `n = 127/128` and
`n = -149/-150` at float64 with `x = 1.0`, `1e-300`, `1e300`, `0.0` and `inf`.

## Impact

`ldexp` is the standard way to build or scale values by exact powers of two (quantisation,
fixed-point packing, `frexp`-style range reduction, probability log-space arithmetic).  For float64
inputs any scale beyond `±2**127` silently saturates: results become `inf` (and immediately `nan` in
the next subtraction) or `0`, so a numerically meaningful range-reduction step destroys data instead
of producing a large-but-finite number.  The failures are silent and the input is entirely ordinary.

## Controls (all in `repro.py`)

* `n = 127` and `n = -149` (inside the float32 range) match NumPy exactly, as do `n = 0, 1, 100`,
  `x = 123.456`, and the signed-zero cases at small `n`.
* `ldexp(1.0, 1024) = inf` and `ldexp(1.0, -1075) = 0` are genuine float64 overflow/underflow and
  are correct.
* The **float32 lane** matches NumPy at `n = 127, 128, -149, -150`.
* Vector input fails elementwise; `ms.jit` fails identically.

## Novelty

Searched the MindSpore tracker for `ldexp`, `ops.ldexp overflow`, `ldexp wrong`, `ldexp inf`: no
report of any `ldexp` defect exists.  The campaign's earlier MindSpore precision findings are other
operators (MS-011 float32 kernels for float64 activations, MS-009 the PYNATIVE subnormal flush);
this one produces `inf`/`0`/`nan` for *normal, finite, representable* results and has an exact
float32-range boundary rather than a warm-up or subnormal mechanism.

## Environment

MindSpore 2.9.0 CPU wheel, NumPy 1.26.4, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; boundary sweep in `evidence.txt`.

---

## Anticipated objections and responses

* **"`ldexp` with a huge exponent is an unusual call."**  The failing results are not only at huge
  exponents: `ldexp(1e-300, 1023)` should be `8.99e+07` and `ldexp(1e300, -150)` should be
  `7.0e+262` - ordinary magnitudes produced by an out-of-float32 *scale*.  Any code that compensates
  a large exponent in `x` with an opposite exponent in `n` (range reduction, log-space arithmetic)
  hits it.
* **"float32 precision is documented for CPU kernels."**  Not for `ops.ldexp` on float64 inputs: the
  result dtype is float64 and every other framework returns the float64 value; the float32 lane is
  separately correct, so this is a dtype dispatch error, not a documented precision bound.
* **"Is this just the known subnormal flush?"**  No: `ldexp(1.0, 128)` is a *normal* float64 value
  and the float32 lane handles the same boundary correctly.
