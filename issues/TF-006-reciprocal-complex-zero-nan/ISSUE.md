# `tf.math.reciprocal` on a complex zero returns `nan+nanj` instead of `inf+nanj` on the vectorised path, so the result depends on the tensor length

**Component:** TensorFlow 2.21.0, CPU eager, `tf.math.reciprocal` on `complex64`/`complex128`
**Severity:** wrong *class* of result — `nan` where C99, NumPy and TensorFlow's own XLA produce a
signed `inf`
**Status:** present in tensorflow 2.21.0 (CPU, AVX2).  XLA (`jit_compile=True`) is correct at every
length, so TensorFlow's two implementations of the same function disagree with each other
**Found with:** Rondo's complex branch-cut probe
(`artifact/hunt-v2/tools_complex_branch_probe.py`, fixed grid plus a length sweep); the certified
reproducer is `repro.py` here (`--replays 2` -> `RESULT: 0 failing check(s).`)

**Nightly follow-up (2026-09-27):** `2.23.0-dev20260925` still produces length-dependent
results, but full packets now return `inf-infj` while tails return `inf+nanj`.
See [Nightly verification and updated issue text](NIGHTLY.md) for two fresh-process replays.

---

## Summary

`1/(0+0j)` is `+inf+nanj` under C99 Annex G.  NumPy returns that, and so does TensorFlow's XLA path
at every length.  The CPU **eager** kernel returns `nan+nanj` instead — but only for elements that are
part of a full Eigen packet, four elements for `complex64` and two for `complex128`.  Elements in the
tail, and tensors shorter than one packet, are correct:

```python
import numpy as np, tensorflow as tf
z = tf.constant([complex(0.0, 0.0)] * 8, dtype=tf.complex64)
print([complex(v) for v in np.asarray(tf.math.reciprocal(z))])
```

```text
eager, complex64, n=1..3 : inf+nanj          (correct)
eager, complex64, n=4    : nan+nanj  x4      <-- wrong class
eager, complex64, n=5    : nan+nanj  x4, inf+nanj   (packet wrong, tail right)
eager, complex64, n=8    : nan+nanj  x8
eager, complex64, n=9    : nan+nanj  x8, inf+nanj
XLA,   every length      : inf+nanj          (correct)
numpy, every length      : inf+nanj
```

`complex128` shows the same pattern with a packet of two (`n=1` correct, `n=2` wrong, `n=3` packet
wrong and tail correct, …).

## Reproduction

```bash
python repro.py --replays 2      # RESULT: 0 failing check(s).
```

The reproducer asserts the packet/tail pattern for both dtypes, the XLA result at every length, the
exact wrong and right values, and a control with a non-zero divisor (`1/(1+0.5j)` matches NumPy) so
that "only the zero divisor is affected" is checked rather than assumed.

## Why this is a defect and not a convention

* **TensorFlow disagrees with itself.**  `tf.function(lambda t: tf.math.reciprocal(t),
  jit_compile=True)` returns `inf+nanj` for every length; only the eager kernel returns `nan`.
* **The boundary is exactly the Eigen packet width**: correct below one packet and in the tail,
  wrong inside full packets — the signature of two formulas in one kernel (the vectorised complex
  division `(a*c + b*d)/(c*c + d*d)` evaluates `0/0` at a zero divisor, whereas the scalar path
  special-cases it).
* **`nan` versus `inf` is observable**: `tf.math.is_finite`, `is_inf`, `nan` propagation in
  downstream arithmetic and comparisons all behave differently, and no tolerance check distinguishes
  them.

## Adjacent reports (searched before writing this)

* [tensorflow#48350](https://github.com/tensorflow/tensorflow/issues/48350) — *"divide_no_nan gives
  nans for small complex values in non-singleton tensors"* (2021, **closed** in 2021-04-27).  Same
  family — a shape-dependent complex `nan` in a division-like op — but a different op
  (`divide_no_nan`, whose contract is to suppress the NaN), different inputs (small non-zero complex
  values, not an exact zero divisor), and closed years before this build.
* [tensorflow#111501](https://github.com/tensorflow/tensorflow/issues/111501) — *"reciprocal_no_nan
  returns inconsistent results for identical complex `inf+0j` inputs"* (**open**).  Again the
  `_no_nan` variant, and about an `inf` divisor rather than a zero one.
* No report was found for `tf.math.reciprocal` itself returning `nan` for a zero complex divisor, nor
  for the length dependence of that result.  Searches: `complex division by zero`,
  `reciprocal nan inf complex`, `complex divide nan`, `1/0 complex inf nan`, plus the earlier
  `complex branch cut` / `complex signed zero` / `tf.math.sqrt complex` set.

Related work in this campaign (not duplicates): `TF-005` (this repository) is the same
vectorised-versus-scalar mechanism in the same kernel family, but for `tf.math.sqrt` and the complex
branch cut (an order-1 wrong sign rather than a `nan`/`inf` class change).

## Proposed fix sketch

Route the tail and sub-packet elements through the same path as full packets, or make the vectorised
complex reciprocal use the scalar formula for a zero divisor, so that `1/(0+0j)` is `inf+nanj`
regardless of the number of elements.  The `complex128` (packet 2) and `complex64` (packet 4) cases
should both be covered by a regression test that varies the tensor length.
