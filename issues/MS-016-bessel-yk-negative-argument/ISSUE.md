# `ops.bessel_y0`/`bessel_y1`/`bessel_k0`/`bessel_k1` silently clamp every negative argument to the `x -> 0` limit (`-inf`/`+inf`) instead of returning `NaN`

**Component:** MindSpore 2.9.0 CPU, `mindspore.ops.bessel_y0`, `bessel_y1`, `bessel_k0`, `bessel_k1`,
float64 and float32, PYNATIVE_MODE and GRAPH_MODE (`ms.jit`)
**Severity:** silent wrong value over the whole negative half-line. `bessel_y0(-1.0)` returns `-inf`
and `bessel_k0(-1.0)` returns `+inf` where the function is not real-valued and SciPy returns `NaN`.
The `inf` is not detectable as an error: `ops.exp(bessel_y0(-1.0))` quietly returns `0.0` instead of
propagating a `NaN`.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** a MindSpore special-function boundary differential against SciPy
(`artifact/hunt-v2/tools_ms_special_probe.py`), then minimised to the negative axis with the
standalone probe shipped next to this file.

---

## Summary

`Y_n(x)` and `K_n(x)` (Bessel functions of the second kind, ordinary and modified) are defined for
`x > 0`.  For `x < 0` no real-valued branch exists, and the reference implementations report that
domain error as `NaN` (SciPy 1.17.1: `y0(-1.0) = nan`, `k0(-1.0) = nan`).  MindSpore 2.9.0 instead
returns the value of the **`x -> 0+` limit** for *every* negative input, including `-1e-300`:

| call (float64) | MindSpore 2.9.0 | SciPy 1.17.1 |
|---|---|---|
| `bessel_y0(-1e-300)` | `-inf` | `nan` |
| `bessel_y0(-1e-8)`   | `-inf` | `nan` |
| `bessel_y0(-1.0)`    | `-inf` | `nan` |
| `bessel_y0(-2.5)`    | `-inf` | `nan` |
| `bessel_y0(-10.0)`   | `-inf` | `nan` |
| `bessel_y0(-100.0)`  | `-inf` | `nan` |
| `bessel_y1(-1.0)`    | `-inf` | `nan` |
| `bessel_k0(-1e-300)` | `+inf` | `nan` |
| `bessel_k0(-1.0)`    | `+inf` | `nan` |
| `bessel_k1(-10.0)`   | `+inf` | `nan` |

The behaviour is a clamp, not a rounding problem: the result is exactly `-inf` for every negative
float64 tested (subnormal `-1e-300`, tiny `-1e-8`, normal `-1.0`/`-2.5`, large `-10.0`/`-100.0`) and
exactly `+inf` for the modified functions, i.e. the value of the limit at zero.  The `x == 0` point itself is correct and is **not**
part of this report: `y0(+/-0.0) = y1(+/-0.0) = -inf` and `k0(+/-0.0) = k1(+/-0.0) = +inf`, matching
SciPy exactly (the separate upstream report mindspore#389 concerns `bessel_y1` at zero only).

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(165 checks per replay, 2 fresh processes.)

```python
import numpy as np, mindspore as ms
from mindspore import ops
import scipy.special as sp
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

x = ms.Tensor(np.asarray([-1.0], np.float64), ms.float64)
float(ops.bessel_y0(x).asnumpy()[0])   # -inf   <-- should be nan
float(sp.y0(-1.0))                     # nan
float(ops.bessel_k0(x).asnumpy()[0])   # +inf   <-- should be nan
float(ops.exp(ops.bessel_y0(x)).asnumpy()[0])  # 0.0  <-- the error is silent
```

What the reproducer pins:

| dimension | result |
|---|---|
| cells | 4 second-kind functions x 6 negative float64 arguments (`-1e-300`, `-1e-8`, `-1.0`, `-2.5`, `-10.0`, `-100.0`), each pinned as the exact `+/-inf` sentinel, as `isnan == False`, as `isinf == True`, and against SciPy's `nan` |
| float32 lane | `bessel_y0/y1` return `-inf` for `-1.0`, `-1e-8`, `-1e-30` where SciPy returns `nan` |
| graph lane | a retained `ms.jit` callable returns `[-inf, -inf, -inf]` for `bessel_y0([-1.0, -1e-8, -1e-300])` and `[inf, inf, inf]` for `bessel_k0`, on two consecutive calls |
| consequence | `exp(bessel_y0(-1.0)) = 0.0` (a plausible value) whereas `exp(nan) = nan` (detectable) |
| controls | `bessel_j0(-1.0) = 0.7651976865579665` and `bessel_i0(-1.0) = 1.2660658777520082` match SciPy exactly on the same negative inputs; `y0/y1/k0/k1` at `0.5`, `1.0`, `2.0`, `4.0` match SciPy to 1 ulp; all four zero points (`+/-0.0`) match SciPy exactly |

The first-kind functions are correct on the negative axis, so the defect is confined to the
second-kind kernels - exactly the ones whose real branch stops at zero.

## Root cause (pointer)

The CPU second-kind kernels take the magnitude/absolute value (or otherwise clamp the argument to
`[0, inf)`) before evaluating the positive-argument routine, so a negative input is served the
`x -> 0` limit instead of a domain error.  The fix is to gate the negative branch to `NaN` (as SciPy
does) or to raise for real inputs; a regression test is
`isnan(ops.bessel_y0(x))` and `isnan(ops.bessel_k0(x))` for `x in {-1e-300, -1.0, -100.0}`, next to
the existing zero-point expectations.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `bessel`, `bessel_y0`, `bessel y negative`,
  `bessel_k0`, `bessel i0` returns only **mindspore#398 / #397** (`ops.bessel_i0` overflow at `713.0`),
  **mindspore#389** (`ops.bessel_y1` returns `NaN` at *zero* instead of `-inf`) and the unrelated
  Gitee implementation tasks.  None of them is about the negative half-line; #389 is the opposite
  point and the opposite error, and #398 is an overflow on the positive axis.
* A web search for the negative-argument behaviour surfaces only the API documentation pages
  ([bessel_y0 r2.9.0](https://www.mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.bessel_y0.html)),
  which document the standard second-kind formula and no negative-input restriction.
* The campaign's other MindSpore drafts cover different operators (`fmod`/`remainder` zero signs,
  `expm1`/`log1p` cancellation, `polygamma`, `softplus`, `mint` kernel registration), so this is a
  new site.
