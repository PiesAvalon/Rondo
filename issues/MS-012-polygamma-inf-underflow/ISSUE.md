# `ops.polygamma(n, +inf)` returns NaN for every order, and the large-argument asymptotic underflows to `0.0`

**Component:** MindSpore 2.9.0, CPU, `mindspore.ops.polygamma` (eager and `ms.jit`), float64
**Severity:** silent wrong value at a point where the function has a well-defined finite limit
(`psi^(n)(+inf) = 0`), plus a normal double replaced by zero at large finite arguments.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** a MindSpore special-function vs SciPy 1.17 differential over the boundary grid
(`artifact/hunt-v2/tools_ms_special_probe.py`, 468 evaluations), minimised to the large-argument
branch.

---

## Summary

For large `x` the polygamma function has the asymptotic series

```
psi^(n)(x) ~ (-1)^(n-1) (n-1)! / x^n        (x -> +inf, n >= 1)
```

so `psi^(n)(+inf) = 0` with the sign `(-1)^(n-1)`, and `psi'(1e300) = 1e-300` / `psi'''(1e100) = 2e-300`
are **normal** float64 numbers (DBL_MIN is 2.2e-308).  SciPy returns those values; MindSpore returns

| call | MindSpore | SciPy 1.17 | note |
|---|---|---|---|
| `polygamma(1, +inf)` | `nan` | `0.0` | limit is 0 |
| `polygamma(2, +inf)` | `nan` | `-0.0` | limit is 0, sign `(-1)^(n-1)` |
| `polygamma(3, +inf)` | `nan` | `0.0` | |
| `polygamma(4, +inf)` | `nan` | `-0.0` | |
| `polygamma(1, 1e300)` | `0.0` | `1e-300` | normal double replaced by zero |
| `polygamma(3, 1e100)` | `0.0` | `2e-300` | normal double replaced by zero |
| `polygamma(1, 1e100)` | `1e-100` | `1e-100` | control - large finite values below ~1e154 are right |
| `polygamma(n, 0)` | `±inf` | `±inf` | control |

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(60 checks per replay: every order at +inf, the large-finite rows, the asymptotic series, and ~25
control cells.)

```python
import numpy as np, mindspore as ms
from mindspore import ops
import scipy.special as sp

n = ms.Tensor(np.int32(1), ms.int32)
ops.polygamma(n, ms.Tensor(np.float64(np.inf)))   # nan   <-- should be 0.0
sp.polygamma(1, np.inf)                            # 0.0
ops.polygamma(n, ms.Tensor(np.float64(1e300)))     # 0.0   <-- should be 1e-300
```

What the reproducer pins:

| dimension | result |
|---|---|
| orders | `n = 1, 2, 3, 4` all return `nan` at `+inf`; SciPy returns a signed zero with sign `(-1)^(n-1)` |
| large finite | `1e300`/`1e100` for `n = 1` and `n = 3`, with the asymptotic series `(n-1)!/x^n` as an independent reference; `polygamma(1, 1e100) = 1e-100` is a control that shows the branch is right below ~1e154 |
| magnitude | the lost values `1e-300` and `2e-300` are above `np.finfo(float64).tiny`, so this is not the documented subnormal policy |
| lanes | `ms.jit` returns the same `nan`; the vector lane gives three NaNs |
| dtype | the float32 lane returns `nan` at `1e100` while the float64 lane returns `1e-100`, so this is **not** the float32-dispatch family (MS-011) |
| controls | `polygamma(n, x)` for `x = 0.5, 1, 2, 10, 1e10` and `n = 1..4` match SciPy to 1e-12 relative; `polygamma(n, 0) = ±inf` in both |

## Root cause (pointer)

The large-argument branch of the `Polygamma` CPU kernel (the asymptotic series in `1/x`) neither
handles `x = +inf` - where the series evaluates `inf * 0` and yields NaN instead of the limit 0 - nor
preserves the leading `1/x^n` term for `x` beyond ~1e154, where an intermediate power overflows and
the whole expression collapses to `0.0`.  A guard for non-finite `x` (returning a signed zero) plus a
reciprocal formulation (`(n-1)!/x^n` computed as a product of `1/x` factors rather than as a quotient
of two potentially overflowing quantities) covers both.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `polygamma`, `polygamma inf`, `polygamma nan`
  and `ops.polygamma` returns **no issue at all**; a web search surfaces only an old Gitee AICPU
  implementation task, not a bug report.
* The nearest MindSpore special-function issues are different operators and different regimes:
  [#400](https://github.com/mindspore-ai/mindspore/issues/400) `ops.digamma` overflowing near zero,
  [#402](https://github.com/mindspore-ai/mindspore/issues/402) `ops.igamma(100, 100)` accuracy,
  [#389](https://github.com/mindspore-ai/mindspore/issues/389) `ops.bessel_y1` at zero,
  [#397](https://github.com/mindspore-ai/mindspore/issues/397)/[#398](https://github.com/mindspore-ai/mindspore/issues/398)
  `ops.i0`/`ops.bessel_i0` overflowing at 713, and
  [#354](https://github.com/mindspore-ai/mindspore/issues/354) `erfinv` float64 precision.
* The same sweep found `ops.digamma` computed in float32 for float64 inputs (ten results exactly
  float32-representable, verified with Rondo's `check_dtype_dispatch`), but that is the same root
  cause as the filed #400 and is therefore recorded as a known issue rather than drafted; the
  Bessel `Y`/`K` family returning `±inf` for negative arguments where SciPy returns `nan` is recorded
  as a candidate.
