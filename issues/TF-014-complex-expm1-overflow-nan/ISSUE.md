# `tf.math.expm1` on complex inputs returns `nan` in the real part whenever `exp` overflows

**Component:** TensorFlow 2.21.0, CPU eager and `tf.function` (non-XLA), `tf.math.expm1` on
`complex64`/`complex128`
**Severity:** wrong value — the real part is `nan` where the mathematical result (and the result of
`tf.math.exp(z) - 1`, of XLA and of NumPy) is `+inf`; the `nan` then propagates through everything
downstream
**Status:** present in tensorflow 2.21.0 (CPU, AVX2); XLA (`jit_compile=True`),
`tf.math.exp(z) - 1`, the real-dtype `expm1` and NumPy are all correct
**Found with:** the wave-163 complex special-value grid (`artifact/hunt-v2/results/w163_*`), triaged
by Rondo's `check_consensus` oracle with `cmath`/NumPy as arbiter; the certified reproducer is
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

`tf.math.expm1` on complex inputs produces `nan` in the **real component** for every input whose
real part overflows `exp`, while the imaginary component is still computed correctly:

```python
import numpy as np, tensorflow as tf

z = tf.constant(np.complex128(complex(1e308, 1.0)))
print(tf.math.expm1(z).numpy())            # nan+infj      <- wrong
print((tf.math.exp(z) - 1).numpy())        # inf+infj      <- correct
print(np.expm1(np.complex128(z)))          # inf+infj      <- correct
```

```text
                                 tf.math.expm1        numpy / exp(z)-1 / XLA
expm1(1e308+1j)   complex128     nan+infj             inf+infj
expm1(1e308-1j)   complex128     nan-infj             inf-infj
expm1(1e308+0j)   complex128     nan+nanj             inf+nanj   (real part inf)
expm1(inf+1j)     complex128     nan+infj             inf+infj
expm1(inf+0j)     complex128     nan+nanj             inf+0j
expm1(710+1j)     complex128     nan+infj             inf+infj
expm1(90+1j)      complex64      nan+infj             inf+infj
expm1(1e38+1j)    complex64      nan+infj             inf+infj
```

The threshold is not `inf`: **any** finite real part large enough for `exp` to overflow triggers the
bug — `710` in complex128 and `90` in complex64 are ordinary finite numbers.  The defect is silent
(`tf.math.expm1` raises nothing) and it is not a precision issue: the real component flips from a
correct `inf` to `nan`.

## Root cause

The complex `expm1` kernel does not reduce to `exp(z) - 1` before overflowing; the `nan` appears
exactly in the regime where an internal `exp` intermediate is `inf` (an `inf - inf` or a
`0 * inf`), while the imaginary component survives untouched.  The three independent oracles in
`repro.py` pin this down:

* `tf.math.exp(z) - 1` is correct (`inf + inf j`), so `exp` itself is fine;
* XLA's `expm1` is correct (`inf + inf j`);
* the real-dtype `tf.math.expm1(1e308)` is `inf`, so only the complex kernel is affected.

Suggested fix: compute the complex `expm1` in the overflow path as `exp(z) - 1` (or scale
`exp(z/1)` by the magnitude before adding the `-1`), and add a regression test asserting
`isfinite(real) == False and isnan(real) == False` for `expm1(x + iy)` with `x >= 710`
(complex128) / `x >= 90` (complex64).

## Relation to the known complex `log1p` report

tensorflow#80850 ("`tf.math.log1p` and `np.log1p` are inconsistent for complex inputs", closed
unfixed by inactivity in August 2026) reports the *other* member of this family:
`tf.math.log1p(0+inf j)` is `nan+nanj` while NumPy returns `inf+1.57j`.  This report is a different
operator with a different failure: `expm1` loses the **real** component (`nan` instead of `inf`) for
finite overflow-size inputs such as `710+1j`, which `#80850` does not cover.  A fix for one of the
two kernels should be checked against the other, and the two can share a regression test file.

## Controls (all in `repro.py`)

* `expm1(-inf+1j) = -1+0j` (correct), `expm1(1e308j)` (purely imaginary: correct),
  `expm1(709+1j)` (just below the overflow: correct) — the failure is specific to an overflowing
  real part.
* Real `float64` `expm1(1e308) = inf` (correct), `tf.math.exp(z) - 1` (correct), XLA (correct).
* The non-XLA `tf.function` path fails exactly like eager, so this is not eager-only.

## Novelty

Searched the TensorFlow tracker (issues and PRs) for `expm1 complex`, `complex expm1`,
`tf.math.expm1 nan`, `expm1 inf`, `expm1 inconsistent`: the only hits are tensorflow#33192
("Incorrect result when subtracting 1 from exponential of Variable", unrelated autograd report) and
tensorflow#62211 (XLA/JIT disagreement in `raw_ops.Expm1`, unrelated).  No report of the complex
`expm1` infinite-real-part `nan` exists.  The nearest item is tensorflow#80850 (complex `log1p`
with `inf`, closed unfixed), distinguished above.

## Environment

TensorFlow 2.21.0 CPU wheel, NumPy 2.5.1, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64, AVX2/AVX-VNNI.
Full dump in `collect_env.txt`; characterization transcript in `evidence.txt`.

---

## Anticipated objections and responses

* **"This is the same as #80850."**  #80850 is about `tf.math.log1p` and its `nan+nanj` result for
  infinite inputs.  This is `tf.math.expm1`, the real component, and finite overflow-size inputs
  (`710+1j`, `90+1j` in complex64) that the closed issue never mentions; it also has a different
  correct answer (`inf`, not `nan`).  If maintainers prefer, both can be handled in one PR.
* **"`1e308` is an extreme input."**  `710+1j` is not: it is an ordinary finite complex number only
  two units above the `exp` overflow point, and complex64 hits it at `90+1j`.
* **"`nan` and `inf` are both 'not a number'."**  They propagate differently: `nan` poisons every
  subsequent reduction and comparison, while `inf` is the mathematically correct `expm1` limit, so a
  computation that would saturate cleanly turns into a `nan` cascade.
