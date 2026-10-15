# `torch.signal.windows.kaiser` / `torch.kaiser_window` return zero or NaN windows for `beta >= 88.8` in float32

**Component:** PyTorch 2.12.1+cu130, `torch.signal.windows.kaiser` (Python implementation) and
`torch.kaiser_window` (whose **documented default dtype is float32**), through the unscaled
`torch.special.i0`; eager CPU, eager CUDA and `torch.compile` (Inductor).
**Severity:** silent loss of the window.  For every `beta >= 88.8` the float32 Kaiser window is
entirely `0` or `NaN` - the centre, which must be exactly `1.0`, is `NaN`, and every value below the
fp32 `i0` overflow point is `finite/inf = 0`.  `torch.kaiser_window(9, beta=89)` returns an
**all-zero** window with no NaN to flag it; `torch.kaiser_window(9, beta=100)` returns
`[0, 0, 0, nan, nan, nan, nan, 0, 0]`.  The same function in float64 and
`scipy.signal.windows.kaiser` (computed in float64 and cast) return the correct window for every
`beta` up to ~709.
**Status:** present in 2.12.1 in eager CPU/CUDA and under `torch.compile`, i.e. in the shared
arithmetic, not a compiler divergence.  The breakpoint is the float32 overflow of `torch.special.i0`
at `beta = 88.8`; the *true* `I0(88.8) = 1.558362e37` is representable in float32 (`max 3.4e38`), so
the overflow is premature (the exact float32 limit is `beta ~ 91.5`).
**Found with:** the wave-372 Rondo window parity sweep (`torch.signal.windows.*` vs
`scipy.signal.windows.*`, 3564 windows over 24 window variants, 22 lengths, `sym` and two dtypes);
certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`,
168 checks).

---

## Summary

```python
import torch, torch.signal.windows as W

W.kaiser(9, beta=89.0, dtype=torch.float32)
# [0., 0., 0., 0., nan, 0., 0., 0., 0.]        centre must be exactly 1.0
W.kaiser(9, beta=100.0, dtype=torch.float32)
# [0., 0., 0., nan, nan, nan, 0., 0., 0.]
torch.kaiser_window(9, beta=89.0)               # documented default dtype = float32
# [0., 0., 0., 0., 0., 0., 0., 0., 0.]         no NaN, silently degenerate
torch.kaiser_window(9, beta=100.0)
# [0., 0., 0., nan, nan, nan, nan, 0., 0.]

W.kaiser(9, beta=100.0, dtype=torch.float64)    # the same function in float64
# [9.31e-43, 2.43e-15, 1.63e-06, 4.25e-02, 1.0, 4.25e-02, 1.63e-06, 2.43e-15, 9.31e-43]
# scipy.signal.windows.kaiser(9, 100) is the same vector (recorded, scipy 1.18.0)
```

The failure is exactly the premature float32 overflow of the denominator:

| `beta` | `torch.special.i0` fp32 | `torch.special.i0` fp64 | true `I0(beta)` (scipy) | representable in fp32 | `kaiser(9, beta, fp32)` |
|---|---|---|---|---|---|
| 88.0 | `7.034020e36` | `7.034020e36` | `7.034020e36` | yes | correct (`max 1.0`) |
| 88.7 | `1.410861e37` | `1.410861e37` | `1.410861e37` | yes | correct (`max 1.0`) |
| **88.8** | **`inf`** | `1.558362e37` | `1.558362e37` | **yes** | `[0,0,0,0,nan,0,0,0,0]` |
| 89.0 | `inf` | `1.901242e37` | `1.901242e37` | yes | `[0,0,0,0,nan,0,0,0,0]` |
| 90.0 | `inf` | `5.139238e37` | `5.139238e37` | yes | `[0,0,0,nan,nan,nan,0,0,0]` |
| 100.0 | `inf` | `1.073752e42` | `1.073752e42` | fp64 only | `[0,0,0,nan,nan,nan,0,0,0]` |
| 700.0 | `inf` | finite | finite | fp64 only | 7/9 entries non-finite |

`torch.signal.windows.kaiser`'s own float64 path is exact at every beta tested up to 700
(`max |fp64 - i0e reference| <= 1e-15`), and SciPy's float64 window is finite for `beta` up to
~709 - so the window value itself is representable; only the float32 numerator/denominator pair
overflows.

## Root cause

`torch/signal/windows/windows.py::kaiser` (the implementation behind both APIs):

```python
    start = -beta
    constant = 2.0 * beta / (M if not sym else M - 1)
    end = torch.minimum(beta, start + (M - 1) * constant)
    k = torch.linspace(start=start, end=end, steps=M, dtype=dtype, ...)
    return torch.i0(torch.sqrt(beta * beta - torch.pow(k, 2))) / torch.i0(beta)
```

Both Bessel terms are evaluated with the **unscaled** `i0` in the requested dtype, so the
denominator overflows to `inf` as soon as `I0(beta)` exceeds the float32 range - including the whole
interval where `I0(beta)` is still representable (`beta` in `[88.8, ~91.5]`) and, more importantly,
the entire range where the *ratio* is representable but the two factors are not (`beta` up to ~709).
Samples whose numerator is also `inf` (`z >= 88.8`) become `inf/inf = NaN`; all others become
`finite/inf = 0`.

PyTorch already ships the exponentially scaled Bessel function, so the fix is local to the window:

```python
    z = torch.sqrt(torch.clamp(beta * beta - k * k, min=0.0))
    return torch.i0e(z) * torch.exp(z - beta) / torch.i0e(beta)     # I0(z)/I0(beta), stable
```

(`torch.special.i0e(z) = exp(-|z|) I0(z)`, and `z <= beta`, so `exp(z - beta) <= 1`.)  Computing the
window in float64 and casting to the requested dtype - what SciPy and
`tensorflow.signal.kaiser_window` (which uses `bessel_i0e`) do - is an equivalent fix.

## Impact

* `torch.kaiser_window` is documented as `kaiser_window(window_length, periodic=True, beta=12.0,
  dtype=None)` with **float32 as the default dtype**, so the default call path is the broken one:
  `torch.kaiser_window(n, beta=b)` is degenerate zero/NaN for every `b >= 88.8`, with no warning or
  exception.  Kaiser `beta = pi * alpha` with `alpha` up to ~30 (used for very high stopband
  attenuation) lands in the affected range.
* The window is used as an analysis window in STFT/filter design, so the failure propagates as an
  all-zero spectrum or NaN propagation through every downstream computation.
* The breakpoint is version-independent arithmetic (`I0` overflow), and the same expression is used
  on CPU, CUDA and under `torch.compile`, so every PyTorch lane is affected identically.

## Controls (all in `repro.py`, 168 checks, 0 failing on each of 2 replays)

* **Boundary control** - `beta = 88.7` is exact in float32 (matches the i0e reference to 1e-4
  relative) while `beta = 88.8` is dead, for six `(M, sym)` combinations; the exact threshold is the
  fp32 overflow of `i0`, not the window formula.
* **Representability control** - the true `I0(88.8) = 1.558362e37` and `I0(91.5) = 2.284237e38` are
  both below the float32 maximum, so the overflow is premature; `torch.special.i0` in float64
  matches SciPy at every recorded point.
* **Reference control** - the comparison reference is built inside PyTorch from `torch.special.i0e`
  (`i0e(z) * exp(z - beta) / i0e(beta)`, float64), not from another framework's output; recorded
  SciPy values (`kaiser(9, 100)`, `i0`) are cross-checked against it.
* **Float64 control** - the same function with `dtype=torch.float64` matches the reference to 1e-12
  at `beta` = 0, 8.6, 50, 88, 88.7, 88.8, 89, 100, 200, 700, and `beta = 0` is exactly all-ones in
  float32.
* **Lane control** - eager CPU, `torch.compile` (Inductor) and CUDA return bit-identical windows,
  including the NaN positions, so this is the shared arithmetic and not a compiler bridge.
* **Failure-mode control** - `kaiser(8, beta=89, dtype=float32)` is *all zeros* (no NaN at all),
  which is why the defect is silent for many sizes.

## Novelty

Searched the PyTorch tracker on 2026-09-13 (`kaiser nan`, `kaiser float32`, `kaiser_window beta`,
`i0 overflow`): the only kaiser-NaN report is **#115595** (fixed by **#116470**, January 2024), which
is a *different* defect - `kaiser(3, beta=7.8)` returned NaN because `sqrt(beta*beta - k*k)` went
slightly negative in float32, fixed by casting `beta`.  The present report is the **denominator
overflow**: it needs `beta >= 88.8` and survives the #116470 fix, exactly as `sqrt` is no longer the
issue there.  The related open item is **#192293** ("`torch.special.i0/i1` (and `torch.i0`) overflow
to `inf` at large finite float64 inputs (~713.0)") with the proposed fix **#192457**: that is the
same *root-cause class* at the `i0` level for float64, while this report is the float32
**window-level** failure, which (a) starts ~3 units of `beta` earlier than the true fp32 limit,
(b) makes a documented public window function return a degenerate result, and (c) has its own local
fix (`i0e`) that does not depend on #192457 landing.  No campaign record covers
`torch.signal.windows` beyond the wave-368 `hann_window`/`hamming_window` parity checks.

## Environment

```
torch 2.12.1+cu130, numpy 2.5.3, Python 3.12.13, WSL2, CPU + RTX 4090 (CUDA)
references recorded from scipy 1.18.0 (jax/jaxlib 0.10.2 interpreter) - see collect_env.txt and
evidence.txt
```

## Anticipated objections and responses

* **"`beta = 88.8` is an extreme parameter."**  The *window* is representable (all values in
  `[0, 1]`) and is returned correctly by the same function in float64 and by SciPy; the affected
  interval is `beta` in `[88.8, ~709]`, not a single point.  `beta = pi * alpha` reaches 88.8 at
  `alpha = 28.3`, an attenuation extreme but a documented use of the parameter (the docstring gives
  no upper bound and only requires `beta >= 0`).
* **"This is just the `i0` overflow (#192293)."**  Partly - the shared cause is the unscaled `i0`,
  but the fix site and the observable are the window function, which can use `i0e` and be correct
  for every representable `beta` regardless of when `i0` is fixed.  The float32 threshold (88.8)
  is also different from the reported float64 one (~713) and is premature by ~3 in `beta`.
* **"Use `float64`."**  That is the workaround, not a fix: `torch.kaiser_window`'s default dtype is
  float32 and the whole path (including CUDA kernels) is float32 by default.
* **"The window is 'normalized to 1' - maybe returning 0 is a deliberate degenerate case."**  The
  docstring states "The window is normalized to 1 (maximum value is 1)"; an all-zero or
  majority-NaN window contradicts that, and float64 does return a maximum of 1.
