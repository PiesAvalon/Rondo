# `tf.math.sqrt` (CPU eager) loses the sign of a negative-zero imaginary part, so the complex branch cut depends on the tensor length

**Component:** TensorFlow 2.21.0, CPU eager, `tf.math.sqrt` on `complex64`/`complex128`
**Severity:** wrong value — for inputs on the complex branch cut the result is the **conjugate** of
the correct one (e.g. `sqrt(-4-0j)` returns `+2j` instead of `-2j`), and whether it happens depends
only on the number of elements in the tensor
**Status:** present in tensorflow 2.21.0 (CPU, AVX2); XLA gets `sqrt` right, so TensorFlow's two
implementations disagree with each other and with NumPy
**Found with:** Rondo's complex branch-cut probe
(`artifact/hunt-v2/tools_complex_branch_probe.py`) plus a length sweep; the certified reproducer is
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing check(s).`)

---

## Summary

Complex elementary functions have branch cuts, and the sign of a zero imaginary part selects the
side: `sqrt(-1+0j) = +1j` but `sqrt(-1-0j) = -1j`.  NumPy, PyTorch and MindSpore all follow that
convention.  TensorFlow's CPU eager kernel follows it only for elements that fill an Eigen packet —
for `complex64` the packet is four elements, for `complex128` two — and silently returns the other
branch everywhere else:

```python
import numpy as np, tensorflow as tf
point = complex(-1.0, -0.0)

for n in (1, 2, 3, 4, 5, 6, 7, 8):
    x = tf.constant([point] * n, dtype=tf.complex64)
    print(n, [complex(v).imag for v in np.asarray(tf.math.sqrt(x))])
```

```text
1 [1.0]                                   <-- wrong branch (+1j)
2 [1.0, 1.0]                              <-- wrong
3 [1.0, 1.0, 1.0]                         <-- wrong
4 [-1.0, -1.0, -1.0, -1.0]                <-- a full packet, correct
5 [-1.0, -1.0, -1.0, -1.0, 1.0]           <-- tail element wrong
6 [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]      <-- tail wrong
7 [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
8 [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]   <-- two packets, correct
```

`complex128` behaves the same with a packet of two:

```text
n=1 [1.0]            n=2 [-1.0, -1.0] (correct)      n=3 [-1.0, -1.0, 1.0]
```

The input really does carry the negative zero (`np.signbit(tf.constant([point]).numpy().imag[0])` is
`True`), the magnitude is unaffected (`|sqrt(-4-0j)| = 2` on both branches), and the wrong answer is
exactly the conjugate of the correct one.

The same packet-boundary defect appears in a *graph* context too: `tf.function(lambda t:
tf.math.pow(t, 0.5))` returns the upper branch for `-0.0` inputs at n=1..3 and for every `n % 4` tail
element, while eager `pow` and XLA `pow` are both correct.  The reproducer asserts that pattern as
well, so the defect is not exclusive to eager execution.

An additional case of the same sign-dropping shows up even on the packet path: `sqrt(0-0j)` returns
`+0+0j` where NumPy returns `+0-0j`.  Both values are zero, so that instance is cosmetic on its own —
it is recorded because it shows the kernel does not preserve the sign in general, and the reproducer
asserts it.

## Reproduction

```bash
python repro.py --replays 2      # RESULT: 0 failing check(s).
```

The reproducer asserts the length pattern, the input signbit, the XLA result, the magnitude control,
and the NumPy reference — 29 checks, all reproducing as documented (`evidence.txt`).

## Why this looks like a vectorisation issue and not a convention

* **Same framework, same function, different answer.**  `tf.function(lambda t: tf.math.sqrt(t),
  jit_compile=True)` returns the correct lower branch at *every* length (1…9, both dtypes), and plain
  `tf.function` matches eager.  If the upper branch were the intended convention, XLA would not
  disagree with eager.
* **The boundary is exactly the Eigen packet width.**  Correct results start at 4 elements for
  `complex64` and 2 for `complex128`, and every `n % packet` tail element is wrong — the fingerprint
  of a vectorised loop whose scalar remainder uses a different formula.
* **The kernel is registered for complex types.**  `tensorflow/core/kernels/cwise_op_sqrt.cc` calls
  `functor::sqrt` for `complex64`/`complex128`, which forwards to Eigen's `scalar_sqrt_op`; its
  packet path and its scalar path do not agree on the sign of the zero imaginary part.
* **A sibling defect with the roles reversed.**  Under `jit_compile=True` (XLA),
  `tf.math.acosh(-1-0j)` returns `+3.14159j` where the correct value is `-3.14159j` (a difference of
  `2*pi`, asserted in the reproducer), and `tf.math.asin` flips the sign of a zero imaginary part;
  for those two ops eager is the correct implementation.  So neither implementation is uniformly
  right: `sqrt` is eager-wrong/XLA-right, `acosh`/`asin` are XLA-wrong/eager-right.

## Cross-framework reference

`sqrt(-1-0j)` / `sqrt(-4-0j)`, complex64 (see also `tools_complex_branch_probe.py`):

| implementation | `sqrt(-1-0j)` | `sqrt(-4-0j)` | verdict |
|---|---|---|---|
| C99 / NumPy 2.5.1 | `-1j` | `-2j` | reference |
| PyTorch 2.12.1 eager | `-1j` | `-2j` | correct |
| MindSpore 2.9.0 PYNATIVE | `-1j` | `-2j` | correct |
| TensorFlow 2.21.0 eager, n=1 | `+1j` | `+2j` | **wrong** |
| TensorFlow 2.21.0 `jit_compile=True` | `-1j` | `-2j` | correct |
| JAX 0.10.2 (eager and jit) | `+1j` | `+2j` | wrong, tracked upstream as jax#39110 |

JAX's variant of this defect is already reported (`jax-ml/jax#39110`, open, "jax.numpy.sqrt ignores
negative zero imaginary part on complex branch cut", with the same root-cause note that it lives in
XLA's complex-sqrt lowering).  The TensorFlow case below/above is a different code path: it appears in
TensorFlow's **eager** kernel on CPU, it is **length-dependent**, and TensorFlow's XLA is correct
where eager is wrong.

## Impact

Any complex-valued computation that lands on a branch cut — signal processing, quantum simulation,
complex-valued neural networks, eigenvalues of non-Hermitian operators — silently returns the
conjugate result for a subset of elements that depends on the tensor length.  Because the wrong
branch differs from the right one by `2i*Im(sqrt)` (order 1 for ordinary inputs), the error is not a
rounding artifact and will not be caught by a tolerance check.  Batching a computation differently
(say, processing 3 rows instead of 4) can change the answer.

## Novelty assessment

Searches performed before writing this report (2026-09-12) over the TensorFlow tracker:

* `gh search issues --repo tensorflow/tensorflow` for `complex branch cut`, `complex signed zero`,
  `sqrt complex`, `tf.math.sqrt complex negative zero`, `complex sqrt imaginary sign`,
  `branch cut complex`, `vectorized complex sqrt`, `complex asin acosh` — the only related hit is
  [tensorflow#118034](https://github.com/tensorflow/tensorflow/issues/118034) (`tf.math.log` returns
  NaN for huge complex inputs), which is a different defect (`nan` above `sqrt(FLT_MAX)`), and the
  closed [tensorflow#55823](https://github.com/tensorflow/tensorflow/issues/55823) (`Square`/`Sqrt`
  do not support complex values on GPU), which is about missing support, not wrong branch selection.
* No report was found for `tf.math.sqrt` losing the branch cut, for a length-dependent complex result,
  or for the XLA `acosh`/`asin` branch flip.
* The JAX report [jax-ml/jax#39110](https://github.com/jax-ml/jax/issues/39110) (open) covers the same
  *mathematical* problem in a different framework and a different implementation; it is cited here
  rather than treated as covering TensorFlow.

Related but distinct work inside this campaign (not duplicates):

* `TF-003` (this repository) is the same *mechanism* — Eigen packet boundary versus scalar tail —
  for `tf.maximum`/`tf.minimum` and the sign of zero, with packet widths 8 (fp32) and 4 (fp64).  The
  defect reported here is a different op family (complex cwise sqrt), a different observable
  (the branch cut, order-1 value error), and different packet widths (4/2).
* `TF-001` (this repository) is a length-dependent *sign of zero* in `tf.math.xlogy`/`xlog1py`.

## Proposed fix sketch

Make the scalar/tail path of the complex `sqrt` use the same formula as the packet path — the
sign-preserving one, i.e. `imag(sqrt(z)) = copysign(sqrt((|z| - x) / 2), imag(z))` — or route the
remainder through the packet implementation. The same review applies to `asin`/`acosh` on the XLA
side, where the sign of the zero imaginary part is dropped instead.
