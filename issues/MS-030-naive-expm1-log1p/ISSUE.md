# MindSpore's float32 `expm1`, `log1p` and `tanhshrink` are the naive `exp(x) - 1`, `log(1 + x)` and `x - tanh(x)`

**Component:** MindSpore 2.9.0, CPU: `ops.expm1`, `ops.log1p`, `ops.tanhshrink`, `ops.Log1p()`,
`ms.numpy.expm1`, `ms.numpy.log1p`, in `PYNATIVE_MODE`, `GRAPH_MODE` and `ms.jit`, `float32` (and the
same naive form at `float64` for the first two).  Scalar and vector inputs.
**Severity:** silent wrong values with a relative error of **1.0** (an exact `0.0`) over many orders of
magnitude, and up to 19% just above that.  `expm1` and `log1p` exist *precisely* to be accurate for
small arguments, so returning `0.0` for `log1p(1e-8)`, `expm1(1e-8)` or `tanhshrink(1e-4)` defeats
the function: the entire signal is lost, while neighbouring functions (`sin`, `tan`, `asin`, `atan`, `sinh`) are
correct.  Gradients are unaffected, so training that uses these functions in the loss/log-space but
not as activations may go unnoticed.
**Status:** present in MindSpore 2.9.0 on CPU, `float32` and `float64`.  The float64 lane is closer
(the cutoff is at `2^-53` instead of `2^-25`) but the implementation is the same naive form.
**Found with:** the wave-209 tiny-argument invariant scan (MindSpore values against NumPy's correctly
rounded float32 result, plus a bit-for-bit identity check against the naive expression); certified
reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 111 checks).

---

## Summary

```python
import numpy as np, mindspore as ms
from mindspore import ops, Tensor
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

ops.expm1(Tensor([1e-8], ms.float32))          # 0.0      numpy float32: 9.99999993922529e-09
ops.log1p(Tensor([1e-8], ms.float32))          # 0.0      numpy float32: 9.99999993922529e-09
ops.log1p(Tensor([2**-24], ms.float32))        # 0.0      numpy float32: 5.960464477539063e-08

# the mechanism, bit for bit, in float32:
ops.expm1(Tensor([1e-6], ms.float32)) == ops.exp(Tensor([1e-6], ms.float32)) - 1.0     # True
ops.log1p(Tensor([1e-7], ms.float32)) == ops.log(Tensor([1.0 + np.float32(1e-7)], ms.float32))
                                                                                       # True
```

`expm1`, float32 (NumPy's float32 result is the correctly rounded reference):

| x | MindSpore | NumPy float32 | relative error |
|---|---|---|---|
| 1e-8 | **0.0** | 9.99999993922529e-09 | **1.0** |
| -1e-8 | **0.0** | -9.99999993922529e-09 | **1.0** |
| 1e-7 | 1.1920928955078125e-07 | 1.000000082740371e-07 | 0.1921 |
| 1e-6 | 9.5367431640625e-07 | 1.0000004522225936e-06 | 0.0463 |
| -1e-6 | -1.0132789611816406e-06 | -9.999995427278918e-07 | 0.0133 |
| 1e-5 | 1.0013580322265625e-05 | 1.0000049769587349e-05 | 1.36e-03 |
| 1e-4 | 1.0001659393310547e-04 | 1.0000499605666846e-04 | 1.16e-04 |

`log1p`, float32:

| x | MindSpore | NumPy float32 | relative error |
|---|---|---|---|
| 1e-8 | **0.0** | 9.99999993922529e-09 | **1.0** |
| -1e-8 | **0.0** | -9.99999993922529e-09 | **1.0** |
| 2^-25 = 2.98e-08 | **0.0** | 2.9802322387695312e-08 | **1.0** |
| 2^-24 = 5.96e-08 | **0.0** | 5.9604644775390625e-08 | **1.0** |
| 1e-7 | 1.1920928244535389e-07 | 9.999999406318238e-08 | 0.1921 |
| 1e-6 | 9.536738616588991e-07 | 9.999995427278918e-07 | 0.0463 |
| 1e-5 | 1.0013530300057027e-05 | 9.999949725170154e-06 | 1.36e-03 |

`tanhshrink`, float32 (`x - tanh(x) ~ -x^3/3`; the reference is the float64 value of the
float32-rounded input, and the exact value is representable at every row):

| x | MindSpore float32 | exact `x - tanh(x)` | relative error |
|---|---|---|---|
| 1e-3 | 3.4924596548080444e-10 | 3.3333324989115454e-10 | 4.8e-02 |
| 5e-4 | 5.820766091346741e-11 | 4.1666670114182125e-11 | 3.97e-01 |
| **1e-4** | **0.0** | 3.3333329586132665e-13 | **1.0** |
| **1e-5** | **0.0** | 3.333328757329848e-16 | **1.0** |
| **1e-6** | **0.0** | 3.333074647445672e-19 | **1.0** |
| **-1e-4** | **0.0** | -3.3333329586132665e-13 | **1.0** |
| 0.5 (control) | 3.7882834672927856e-02 | 3.788284212350845e-02 | 2.0e-07 |
| 1.0 (control) | 2.3840582370758057e-01 | 2.3840583860874176e-01 | 6.3e-08 |

so `tanhshrink` in float32 loses its entire value below `1e-4`, is 40% off at `5e-4`, and its
gradient (`tanh(x)^2`) is exactly `0.0` at `x = 1e-4` where the exact derivative is `1.0e-08`.
The float64 lane is accurate (`|ms - exact| <= 1e-12` relative at `1e-4`).

The zero region is exactly the float32 rounding of `1 + x`: `log1p`/`expm1` return 0 whenever
`1 + x` (or `exp(x)`) rounds to 1, i.e. for `|x| < 2^-25` in `log1p`'s case and for `|x|` below the
`exp` cutoff in `expm1`'s.  Above it, the result jumps in steps of `2^-23` (the float32 spacing near
1), which is why the relative error is 19% at `1e-7` and 4.6% at `1e-6`.

## Root cause

Both kernels evaluate the naive expression:

* `ops.expm1(x)` is bit-for-bit `ops.exp(x) - 1` in float32 for every tested point (`1e-8` ... `1e-3`,
  both signs);
* `ops.log1p(x)` is bit-for-bit `ops.log(1 + x)` in float32 for every tested point (`2^-25`, `2^-24`,
  `1e-8` ... `1e-4`, both signs);
* `ops.tanhshrink(x)` is bit-for-bit `x - tanh(x)` in float32 for every tested point (`1e-3` ... `1e-6`
  and `-1e-4`).

The subtraction/addition of 1 is exact only when the tiny term survives the rounding of
`1 + x`/`exp(x)`, so the significant digits are destroyed exactly in the range the functions are
meant to serve.  The `float64` lane uses the same form (its cutoff is `2^-53`): `ops.log1p(1e-20)` is
`0.0` in float64 as well, and `ops.expm1(1e-8)` in float64 has a relative error of `6.6e-08` against
the exact `1.0000000050000001e-08` where NumPy is correctly rounded to the last bit.

This is specific to these two kernels: at `x = 1e-8` in float32, `ops.sin`, `ops.tan`, `ops.asin`,
`ops.atan` and `ops.sinh` all match NumPy exactly (relative error `< 1e-6`, in fact `0`), and
`ops.softplus`/`ops.sqrt`/`ops.erf` show no comparable error.  The gradients are also correct
(`grad(expm1) = exp(x)`, `grad(log1p) = 1/(1+x)`, both within `1e-6` of the exact value at `1e-8`),
so only the forward values are affected.

## Impact

* `log1p` is the standard way to accumulate tiny probabilities or ratios (`log1p(-p)`,
  `log1p(exp(-x))`, log-likelihood increments) and `expm1` for small rates and for
  `expm1(x)/x`-style normalisations.  With this kernel, every `|x| < 2^-25` contributes **nothing**
  (`0.0`) and everything up to `~1e-5` is wrong in the third significant digit or worse.
* The error is silent and systematic: a sum of many `log1p(1e-8)` terms returns exactly 0 instead of
  the accumulated value, and `expm1(1e-6)/1e-6` returns `0.9537` instead of `1.0000` (4.6% off).
* `tanhshrink` is the regulariser `x - tanh(x)`: in float32 every `|x| <= 1e-4` contributes exactly
  nothing and at `5e-4` it is 40% too large, and its gradient is `0.0` at `1e-4` (exact `1e-08`), so
  both the penalty and its derivative vanish for the small weights the term exists to shrink.
* Because the `expm1`/`log1p` gradients are correct, a numerical-gradient check of those passes while
  the forward value is wrong, and because the zero region is an interval rather than a point, it
  cannot be dismissed as a single rounding.

## Suggested fix

Implement both kernels with the compensated forms (C99 `log1p`/`expm1`, `std::log1p`/`std::expm1`, or
the standard `k = 1 + x; log(k) - (k - 1 - x)/k` and Kahan `expm1` algorithms) instead of the naive
expressions, and add regression tests at `1e-8`/`1e-7`/`1e-6` in float32 and float64.  The
reproducer's checks are already in that form, e.g.

```python
v = ms.Tensor([1e-8], ms.float32)
assert float(ops.log1p(v).asnumpy()[0]) != 0.0                  # currently 0.0
assert abs(float(ops.log1p(v).asnumpy()[0]) - 1e-8) < 1e-14     # currently 1e-8 off
```

## Controls (all in `repro.py`, 111 checks, 0 failing on each of 2 replays)

* NumPy's float32 results are the reference at every point (correctly rounded), and the relative
  errors are asserted to the documented values with a 2% tolerance.
* The zero region is asserted directly (`0.0` where the reference is nonzero) at `1e-8`, `-1e-8`,
  `2^-25` and `2^-24`.
* The mechanism is asserted bit-for-bit: `ops.expm1(x) == ops.exp(x) - 1` and
  `ops.log1p(x) == ops.log(1 + x)` over nine and eight float32 points respectively.
* Neighbouring tiny-argument functions (`sin`, `tan`, `asin`, `atan`, `sinh` at `1e-8`) are checked to
  match NumPy, so the defect is specific to `expm1`/`log1p` and not a general small-argument issue.
* Gradients are checked to be correct at `1e-8` (`exp(x)` and `1/(1+x)`), isolating the forward.
* `ops.expm1`/`ops.log1p`, `ops.Log1p()`, `ms.numpy.expm1`/`ms.numpy.log1p`, a vector input,
  `ms.jit` and `GRAPH_MODE` all show the same value; the float64 lane is checked to be the same naive
  form for `expm1`/`log1p` (`log1p(1e-20) == 0.0`).
* For `tanhshrink` the bit-for-bit identity with `x - tanh(x)` is asserted at six points, the
  controls at `0.5`/`1.0` are checked to be within `1e-6` of the exact value, the float64 lane is
  checked to be accurate to `1e-12` relative, the gradient is checked to be exactly `0.0` at `1e-4`
  and nonzero at `1e-3`, and a vector input, `ms.jit` and `GRAPH_MODE` are checked to agree.

## Novelty

MindSpore's tracker is on Gitee.  Searching it for `log1p`, `log1p 精度`, `ops.log1p` and `expm1`
returns six log1p items, all closed, and the nearest ones are **different observables**:

* [IBOIZS](https://gitee.com/mindspore/mindspore/issues/IBOIZS) - `mint.special.log1p` **near -1**
  (input `-0.9999999999...`) returns a huge finite value instead of `-inf`, on **Ascend 910B** and
  through the `mint` API (closed);
* [IBOJ44](https://gitee.com/mindspore/mindspore/issues/IBOJ44) - `log1p` for inputs **below -1**
  returns an error value instead of NaN/domain handling (closed);
* [I69NVQ](https://gitee.com/mindspore/mindspore/issues/I69NVQ), I5EWHH, I48O73, I5HSYU - dtype
  support and interface/performance work for the Log1p operator, not accuracy.

No item describes the tiny-argument precision reported here (an exact `0.0` for `0 < x < 2^-25`, 19%
relative error at `1e-7`, and the bitwise identity with `log(1 + x)`), and none mentions `expm1` or
`tanhshrink`.  A Gitee search for `tanhshrink` returns only CI/precision items for other operators,
and the campaign's own two `tanhshrink` records are negative results from boundary probes
(`-0.0, 0.0, +-1e-38, +-1, +-inf, nan` in wave 118c; float64 accuracy in wave 136), i.e. they never
sampled the `|x| <= 1e-4` region where the fp32 value is exactly zero.  The
campaign's other MindSpore drafts cover different operators (MS-009 subnormal flush, MS-011 float64
dispatched to the float32 kernel, MS-029 `tanh` outside `[-1, 1]`).

## Environment

MindSpore 2.9.0, CPU, Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64, `PYNATIVE_MODE` and
`GRAPH_MODE`.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Float32 has limited precision near 1; this is expected."**  The correctly rounded float32 value
  is representable and NumPy returns it: `log1p(1e-8) = 9.99999994e-09`, not `0`.  Returning `0.0`
  is not a rounding of the true value, it is the loss of the entire result.
* **"`expm1`/`log1p` are niche."**  They are the standard tools for exactly this range, and the
  failure is an interval of exact zeros, not an edge point: every `|x| < 2^-25` contributes nothing.
* **"Use float64."**  The float64 lane is the same naive form: `ops.log1p(1e-20)` is `0.0` there too,
  and `ops.expm1(1e-8)` is `6.6e-08` relatively off where NumPy is exact.
* **"The gradients are what matter."**  The gradients are correct, which is why the defect is silent;
  the forward values feed losses and normalisations directly and are wrong by up to 100%.
