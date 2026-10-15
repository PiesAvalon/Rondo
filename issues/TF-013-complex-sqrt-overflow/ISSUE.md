# `tf.math.sqrt` overflows to `inf` on complex inputs whose real part exceeds half the dtype maximum

**Component:** TensorFlow 2.21.0, CPU eager and `tf.function` (non-XLA), `tf.math.sqrt` on
`complex64`/`complex128`
**Severity:** wrong value — a finite, fully representable root (`1e154` for `1e308+0j`) comes back as
`inf`; the failure is silent and turns a bounded computation into an overflow
**Status:** present in tensorflow 2.21.0 (CPU, AVX2); XLA (`jit_compile=True`) gets it right, and
TensorFlow's own `tf.pow(z, 0.5)`, `tf.exp(0.5*tf.math.log(z))` and the real-dtype `sqrt` are right,
so TensorFlow's implementations disagree with each other and with NumPy
**Found with:** the wave-163 complex special-value grid
(`artifact/hunt-v2/results/w163_*`), triaged by Rondo's `check_consensus` oracle with
`cmath`/NumPy as arbiter, then reduced to this witness; the certified reproducer is `repro.py` in
this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

`tf.math.sqrt` on a complex tensor is computed through Eigen's `complex_sqrt`, which forms

```cpp
w = sqrt(0.5 * (|x| + hypot(x, y)))
```

before scaling.  For a real part near the top of the range the sum `|x| + hypot(x, y)` is about
`2*|x|` and overflows to `+inf` **before** the multiplication by `0.5`, so `w` is `inf` even though
`0.5 * (|x| + hypot(x, y))` and the final root are both finite.  The overflow starts at
`|re| > max/2` (`8.988e307` for `complex128`, `1.701e38` for `complex64`) and is independent of the
tensor length, of the sign of the real part and of the imaginary part.

```python
import numpy as np, tensorflow as tf

z = tf.constant(np.complex128(complex(1e308, 0.0)))
print(tf.math.sqrt(z).numpy())        # inf+0j     <- wrong
print(np.sqrt(np.complex128(z)))      # 1e154+0j   <- correct, and representable
```

```text
tf.math.sqrt( 1e308+0j)  ->  inf+0j          numpy: 1e154+0j
tf.math.sqrt(-1e308+0j)  ->  0+infj          numpy: 1e154j
tf.math.sqrt( 9e307+0j)  ->  inf+0j          numpy: 9.486832980505138e+153+0j
tf.math.sqrt( 1e307+0j)  ->  3.1622776601683792e+153+0j   (correct: below max/2)
tf.math.sqrt( 1e308j)    ->  7.071067811865475e+153+7.071067811865475e+153j   (correct: re = 0)
```

## Expected vs actual

| framework | `sqrt(1e308+0j)` (complex128) | `sqrt(-1e308+0j)` | `sqrt(2e38+0j)` (complex64) |
|---|---|---|---|
| NumPy 2.5.1 | `1e154+0j` | `1e154j` | `1.4142135e+19+0j` |
| PyTorch 2.12.1 (CPU) | `1e154+0j` | `1e154j` | `1.4142135e+19+0j` |
| JAX 0.10.2 (CPU) | `1e154+0j` | `1e154j` | `1.4142134e+19+0j` |
| MindSpore 2.9.0 (CPU) | `1e154+0j` | `1e154j` | `1.4142135e+19+0j` |
| **TensorFlow 2.21.0 (CPU)** | **`inf+0j`** | **`0+infj`** | **`inf+0j`** |

The `1e154` result is an ordinary double (`DBL_MAX = 1.797e308`), so this is not a case of the
*result* overflowing; only an intermediate of the formula does.

## Root cause

TensorFlow's CPU `sqrt` for complex types goes through Eigen's `scalar_sqrt_op` /
`numext::sqrt`, whose complex overload calls `complex_sqrt` in
`Eigen/src/Core/MathFunctionsImpl.h`:

```cpp
// Letting w = sqrt(0.5 * (|x| + |z|))
const T w = numext::sqrt(T(0.5) * (numext::abs(x) + numext::hypot(x, y)));
```

`hypot(x, y)` is itself overflow-safe, but the **sum** `numext::abs(x) + numext::hypot(x, y)` is not:
for `x = 1e308`, `y = 0` it is `2e308 = inf`, and the later `T(0.5) *` cannot repair it.  Every
branch below (`w`, `y / (2*w)`, `|y| / (2*w)`) then produces `inf` or `0`.  That also explains the
exact boundary: overflow needs `|x| + hypot(x,y) > max`, i.e. `|x| > max/2` (for small `|y|`), which
is what `repro.py` measures (`0.98 * max/2` is finite, `1.02 * max/2` is `inf`, for both widths).

The XLA path implements complex `sqrt` with its own stable lowering, which is why it is correct.

## Suggested fix

Scale both terms before adding them (mathematically identical, overflow-free):

```cpp
const T w = numext::sqrt(T(0.5) * numext::abs(x) + T(0.5) * numext::hypot(x, y));
```

or equivalently `numext::sqrt(numext::abs(T(0.5) * x) + T(0.5) * numext::hypot(x, y))`.  A
regression test can compare against `std::sqrt(std::complex)` for `re` in
`{1e307, 9e307, 1e308, -1e308, DBL_MAX}` at both widths.

## Impact

Any complex square root whose real part lands in the top half of the format silently becomes `inf`
(and the conjugate axis becomes `inf` on the imaginary component).  Workloads that normalise or
orthonormalise large complex data — QR/Cholesky-style factorisations on an unscaled matrix,
`sqrt`-based normalisation of Fourier or wavefield data, complex `rsqrt`/`linalg.norm` pipelines
built on `sqrt` — turn a bounded result into `inf`/`nan` one operation later.  The failure is
invisible without a reference: `tf.math.sqrt` raises nothing and the input is finite and ordinary.

## Controls (all in `repro.py`)

* `1e307+0j` and `-1e307+0j` (one decade below: correct), `0.98*max/2` (correct) vs `1.02*max/2` (`inf`).
* `1e308j` (purely imaginary, `re = 0`: correct) and `1e308+1.0j` / `1e308-1.0j` (both fail on the real part).
* Lengths 1, 2, 4, 8 all fail identically — this is **not** the vectorized "length >= 4" path of
  tensorflow#118034.
* Real `float64` `sqrt(1e308) = 1e154` (correct), `tf.pow(z, 0.5)` (correct),
  `tf.exp(0.5 * tf.math.log(z))` (correct), XLA `sqrt` (correct), non-XLA `tf.function` (fails).

## Novelty

Searched the TensorFlow tracker (issues and PRs) for `complex sqrt`, `complex_sqrt`,
`Eigen complex sqrt`, `sqrt complex inf`, `sqrt complex128 overflow`, `tf.math.sqrt complex`,
`complex sqrt nan overflow`, `complex64 sqrt`, `sqrt overflow inf`: no report of the complex `sqrt`
overflow was found.

Adjacent items checked and distinguished:

* **tensorflow#118034** — "`tf.math.log` returns NaN for complex inputs with `|z| > sqrt(FLT_MAX)`
  in vectorized tensors" (open).  Different operator (`log`, not `sqrt`), different threshold
  (`sqrt(FLT_MAX) ≈ 1.84e19` vs `max/2 ≈ 8.99e307`), and explicitly **length-dependent** (only
  tensors of length >= 4 fail, scalar and length 1-3 are correct).  This defect is uniform in the
  tensor length and already fires for a scalar.
* **tensorflow#55823** — `Square`/`Sqrt` do not support complex values on GPU (closed); a support
  gap, not an overflow on CPU.
* **This campaign's TF-005** — `tf.math.sqrt` loses the sign of a negative-zero imaginary part, so
  the branch cut depends on the SIMD packet tail.  Different root cause (sign of the zero, `y == 0`
  branch) and the values here have non-zero or exactly-zero imaginary parts that both overflow.
* **TF-002** (forward-mode complex `div` JVP) and **TF-006** (complex reciprocal at zero) are
  different operators.

## Environment

TensorFlow 2.21.0 CPU wheel, NumPy 2.5.1, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64, AVX2/AVX-VNNI.
Full dump in `collect_env.txt`; characterization transcript in `evidence.txt`.

---

## Anticipated objections and responses

* **"`1e308` is an extreme input."**  It is a finite double, not `inf` or `nan`, and the correct
  output `1e154` is a perfectly ordinary double.  The threshold is half of `DBL_MAX`, i.e. the top
  octave of the exponent range, and complex `sqrt` is exactly the operation whose inputs are
  otherwise expected to be near the top of the range (a squared magnitude doubles the exponent).
* **"NumPy also overflows for some large inputs."**  It does not here: NumPy, PyTorch, JAX and
  MindSpore all return `1e154+0j`, and so does TensorFlow's own XLA path, `pow` and `log/exp`
  formulation — so the CPU eager kernel is the outlier.
* **"Eigen owns the code, not TensorFlow."**  TensorFlow ships and registers this path
  (`cwise_op_sqrt.cc` -> Eigen `scalar_sqrt_op`), so the user-visible defect belongs to TensorFlow;
  the fix is small either upstream in Eigen or by using a dedicated kernel, and an XLA-consistent
  regression test protects it.
