# MindSpore's `igamma`/`igammac` lose float64 accuracy as the shape parameter grows

**Component:** MindSpore 2.9.0, CPU: `ops.igamma(a, x)` and `ops.igammac(a, x)` (the regularized
lower/upper incomplete gamma functions), `float64` (the `float32` lane differs, so both are covered).
**Severity:** silent wrong value.  The float64 relative error against `scipy.special.gammainc` grows
with the shape parameter `a` and is structured by region - at `a = 1000` it reaches **1.3e-04**
(about four significant digits lost in a float64 computation), while at `a = 1` it is exact
(1.8e-16).  `ops.igamma` is the inaccurate side for `x` below the branch boundary and `ops.igammac`
above it, so a caller cannot avoid it by choosing the complement.
**Status:** present in MindSpore 2.9.0 on CPU.  The error is not a float32 kernel dispatch: the
results are not float32-representable and the two dtypes differ.
**Found with:** the gamma-family special-function scan (MindSpore vs SciPy at matching arguments) plus
the campaign's tail oracle; certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 28 checks).

---

## Summary

```python
import mindspore as ms, scipy.special as sp
from mindspore import ops, Tensor
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

float(ops.igamma(Tensor([1000.], ms.float64), Tensor([500.], ms.float64)).asnumpy().reshape(-1)[0])
# 3.2987070012280294e-86        scipy: 3.298272797068251e-86     relative error 1.3e-04

float(ops.igamma(Tensor([1.], ms.float64), Tensor([1.], ms.float64)).asnumpy().reshape(-1)[0])
# 0.6321205588285578            scipy: 0.6321205588285577        relative error 1.8e-16
```

| a | x | `ops.igamma` | scipy | relative error | `ops.igammac` relative error |
|---|---|---|---|---|---|
| 1 | 1 | 0.6321205588285578 | 0.6321205588285577 | 1.8e-16 | 3.0e-16 |
| 2 | 0.01 | 4.966791260568457e-05 | 4.966791334026596e-05 | 1.5e-08 | 7.3e-13 |
| 5 | 0.1 | 7.667800790838442e-08 | 7.667801686189327e-08 | 1.2e-07 | 8.9e-15 |
| 10 | 10 | 0.5420701141370888 | 0.5420702855281478 | 3.2e-07 | 3.2e-07 |
| 100 | 100 | 0.5133034878313246 | 0.5132987982791487 | 9.1e-06 | 9.1e-06 |
| **1000** | **500** | **3.2987070012280294e-86** | 3.298272797068251e-86 | **1.3e-04** | **0.0** (exact) |
| 1000 | 1500 | 1.0 | 1.0 | 0.0 (exact) | **1.3e-04** |

## Root cause

The error is the signature of the series/continued-fraction split used to evaluate the incomplete
gamma: at `a = 1000` the branch that covers `x <= a` is the inaccurate one for `igamma`, and the
complementary `igammac` uses that same branch for `x >= a` (where it is `1 - P`), which is why the two
are wrong in complementary regions and both are wrong near the boundary (`a = 100`, `x = 100`:
9.1e-06 each).  The error grows smoothly with `a` - 1.5e-08 at `a = 2`, 1.2e-07 at 5, 3.2e-07 at 10,
1.8e-06 at 50, 9.1e-06 at 100, 2.6e-05 at 200, 4.6e-05 at 500, 1.3e-04 at 1000 - which points at the
series expansion's truncation/convergence criterion rather than at a single special case.

## Impact

* Statistical code that uses the incomplete gamma (chi-square, Poisson and gamma CDFs, p-values,
  survival functions) silently loses up to four significant digits at large `a`, and the loss depends
  on which of the two functions the caller picks.
* The error is smooth and small enough that tolerance-based tests pass, so it survives review; it is
  a pure float64 accuracy defect (SciPy is exact at the same arguments).
* It is not a dtype dispatch: `float32` and `float64` differ, and the float64 values are not
  float32-representable, so the implementation really runs in float64 - it is the algorithm that is
  not converged.

## Suggested fix

Tighten the series branch's convergence criterion (or evaluate it in extended precision for large
`a`) and add accuracy tests across `a`, especially near `x = a`.  A regression test:

```python
import scipy.special as sp
a, x = 1000.0, 500.0
got = float(ops.igamma(ms.Tensor([a], ms.float64), ms.Tensor([x], ms.float64)).asnumpy().reshape(-1)[0])
assert abs(got - sp.gammainc(a, x)) / sp.gammainc(a, x) < 1e-10
```

## Controls (all in `repro.py`, 28 checks, 0 failing on each of 2 replays)

* `a = 1, x = 1` is exact (1.8e-16) and `a = 1000, x = 1500` is exact for `igamma`, so the operator is
  correct in the regions and at the parameters where the branch converges.
* The relative error is asserted to lie inside a measured band for five `(a, x)` rows and to grow
  monotonically with `a`; the two pinned witnesses are asserted to 10% of their measured values.
* The complementary-region structure is asserted directly (`a=1000, x=500`: `igamma` 1.3e-04 while
  `igammac` is exact; `a=1000, x=1500`: the reverse; `a=100, x=100`: both about 9e-06).
* The float64 value is asserted not to be float32-representable and the float32/float64 lanes are
  asserted to differ, ruling out the float32-dispatch class.
* The SciPy reference is computed at runtime for every row, so the report does not depend on pinned
  SciPy constants.

## Novelty

MindSpore's tracker (Gitee) search for `igamma`/`igammac`/`gammainc` returns only closed items with
different observables: IB8372 (infer-shape failures), I6RQ0F/I6YSN9 (CI and dynamic-shape errors),
I5DUGZ (GPU dynamic shape), I5DKWF (`IgammaGrad` on GPU), I56J5Y/I48O7O (feature completion), and one
2022 compatibility-test item I641M1 ("the accuracy of probability igamma test cases is incorrect",
closed, no numbers or reproduction).  None reports the accuracy-versus-`a` growth, the region
structure, or a float64 relative error of 1e-04.  If a maintainer considers I641M1 to cover it, this
report still supplies the quantified curve, the complementary-region explanation and a control suite.

## Cross-framework position (measured in the same week)

The same argument grid against SciPy in the other two frameworks that expose the incomplete gamma
shows that the defect is MindSpore's implementation, not the algorithm:

| framework | `igamma` max relative error (fp64) |
|---|---|
| **MindSpore 2.9.0** | **1.3e-04 at a = 1000** (this report) |
| JAX 0.10.2 (XLA, CPU and CUDA) | 2.3e-13 at a = 200, 1.0e-11 at a = 5000 |
| SciPy (reference) | - |

JAX's `gammainc`/`gammaincc` were audited over the same (a, x) region, including the large-`a`
saturation cases, and stay within 1.1e-11 relative up to `a = 5000`; `betainc` with equal parameters
up to 1e4 stays within 8.6e-13.  So the growth with `a` is specific to MindSpore's series/continued
fraction branches, and the workaround for a MindSpore user is to compute the function through
`scipy`/`jax` (or in float64 via another framework) rather than to accept the loss.

## Environment

MindSpore 2.9.0, CPU, Python 3.12.13, NumPy 2.5.3, SciPy (reference), WSL2 (Linux 6.6.87.2) x86_64,
`PYNATIVE_MODE` (the same values come from `ms.jit`, checked while scanning).  Full dump in
`collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"The values are tiny, so the absolute error is small."**  The report is in relative terms, which
  is what matters for probabilities and p-values: 1.3e-04 relative at `a = 1000` is four digits lost,
  and the `a = 100, x = 100` row (0.513 vs 0.5133) is an O(1) value with a 9e-06 relative error.
* **"This is the float32 lane."**  No: the float64 values are not float32-representable, the two
  dtypes differ, and `a = 1` is exact at 1e-16 in the same float64 lane.
* **"SciPy is the outlier."**  SciPy's values satisfy the defining recurrence and the known
  identities; MindSpore's error is smooth in `a` and structured by the series/continued-fraction
  boundary, which is a truncation signature rather than a reference disagreement.
* **"Already reported in I641M1."**  That closed item has no numbers, no reproduction and no
  operator-level statement; the curve, the region structure and the dtype controls here are new.
