# `tf.signal.hann_window` / `tf.signal.hamming_window` silently ignore `periodic=True` for odd `window_length`

**Component:** TensorFlow 2.21.0 (CPU), `tf.signal.hann_window` and `tf.signal.hamming_window`
(`tensorflow/python/ops/signal/window_ops.py::_raised_cosine_window`), eager, `tf.function`, XLA and
the `tf.signal.stft` default `window_fn`.
**Severity:** silent wrong values for a documented argument.  For every **odd** `window_length`,
`periodic=True` returns the **symmetric** window - bit-identical to `periodic=False` - so the
argument has no effect at all.  For `window_length = 9` the Hann window is returned as
`[0, 0.14645, 0.5, 0.85355, 1, 0.85355, ...]` instead of the periodic
`[0, 0.11698, 0.41318, 0.75, 0.96985, 0.96985, ...]` (max error `0.267`; Hamming `0.245`; at
`window_length = 3` the error is `0.75`).  Even lengths are correct.
**Status:** present in 2.21.0 and at master (`window_ops.py` lines 236/238), in every execution lane
(eager, `tf.function`, XLA, dynamic `window_length`), and masked by the upstream unit test, whose
reference helper makes the same assumption.
**Found with:** the wave-368 Rondo `tf.signal` differential probe (eager / `tf.function` / XLA /
non-constant shapes against closed-form and cross-framework references); certified reproducer
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 183 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf
tf.signal.hann_window(9, periodic=True, dtype=tf.float64).numpy()
# [0.         0.14644661 0.5        0.85355339 1.         0.85355339
#  0.5        0.14644661 0.        ]        <- the SYMMETRIC window (denominator N-1 = 8)
# scipy.signal.windows.hann(9, sym=False)
# [0.         0.11697778 0.41317591 0.75       0.96984631 0.96984631
#  0.75       0.41317591 0.11697778]        <- the periodic window (denominator N = 9)

np.array_equal(tf.signal.hann_window(9, periodic=True,  dtype=tf.float64).numpy(),
               tf.signal.hann_window(9, periodic=False, dtype=tf.float64).numpy())
# True   <- periodic=True is a no-op for odd lengths
```

`periodic=True` and `periodic=False` differ only when `window_length` is **even**:

| `window_length` | parity | `max|hann_window(-, periodic=True) - periodic reference|` | Hamming | `periodic=True` == `periodic=False` |
|---|---|---|---|---|
| 2, 4, 6, 8, 10, 16, 32, 64, 128 | even | `0.0` (exact) | `0.0` | no (correct) |
| 3 | odd | `7.50e-01` | `6.90e-01` | **yes** |
| 5 | odd | `4.05e-01` | `3.72e-01` | **yes** |
| 7 | odd | `3.61e-01` | `3.32e-01` | **yes** |
| 9 | odd | `2.67e-01` | `2.45e-01` | **yes** |
| 11 | odd | `2.26e-01` | `2.08e-01` | **yes** |
| 17 | odd | `1.45e-01` | `1.34e-01` | **yes** |
| 31 | odd | `7.88e-02` | `7.25e-02` | **yes** |
| 63 | odd | `3.85e-02` | `3.54e-02` | **yes** |
| 127 | odd | `1.90e-02` | `1.75e-02` | **yes** |
| 129 | odd | `1.87e-02` | `1.72e-02` | **yes** |

The reference is the standard periodic raised-cosine window

```
w[n] = a - b * cos(2 * pi * n / N),   n = 0 .. N-1
```

i.e. `scipy.signal.windows.hann/hamming(..., sym=False)`, `torch.hann_window/hamming_window(...,
periodic=True)` and MATLAB's `hann(N, 'periodic')` - all of which agree with each other exactly and
with TensorFlow only for even `N`.  The symmetric window uses denominator `N - 1`.

## Root cause

`tensorflow/python/ops/signal/window_ops.py::_raised_cosine_window` (installed 2.21.0, lines
234-236; master lines 237/239):

```python
    periodic = math_ops.cast(
        ops.convert_to_tensor(periodic, dtype=dtypes.bool, name='periodic'),
        dtypes.int32)
    even = 1 - math_ops.mod(window_length, 2)                 # 1 for even N, 0 for odd N
    n = math_ops.cast(window_length + periodic * even - 1, dtype=dtype)
```

`even` multiplies `periodic`, so for odd `window_length` the product is always `0` and the
denominator collapses to `window_length - 1`, the symmetric one.  The `even` term has no other
effect anywhere: for `periodic=False` it is irrelevant.  The intended denominator is
`window_length` for `periodic=True` and `window_length - 1` for `periodic=False` for both
parities, i.e.

```python
    n = math_ops.cast(window_length + periodic - 1, dtype=dtype)
```

with the existing `window_length == 1` early return unchanged.  `_raised_cosine_window` is shared
by `hann_window` and `hamming_window` only (`kaiser_window`, `kaiser_bessel_derived_window` and
`vorbis_window` do not take `periodic`), which is exactly the set of functions affected.

## Impact

* `tf.signal.stft` defaults to `window_fn = hann_window` and calls it internally as
  `window = window_fn(frame_length, dtype=...)`
  (`tensorflow/python/ops/signal/spectral_ops.py:87`), i.e. with the default `periodic=True`.  Every
  **odd** `frame_length` therefore gets a symmetric analysis window, which is the wrong window for
  an overlapped STFT: for the signal `0..8` with `frame_length = frame_step = 9` and
  `fft_length = 16`, TensorFlow returns a DC bin of `16.0` where the periodic reference gives
  `20.25` (`max|d| = 4.25` over the frame), and every other bin differs too.
* Any user comparing TensorFlow windows/spectrograms with `scipy`, `librosa`-style analyzers or
  `torch.stft` gets a silently different window for odd frame lengths; the windows are smooth and
  plausible, with no error or NaN.
* `tf.signal.hann_window(n, periodic=True)` being a no-op for odd `n` is a direct contradiction of
  the documented argument: *"periodic: A bool `Tensor` indicating whether to generate a periodic or
  symmetric window. Periodic windows are typically used for spectral analysis while symmetric
  windows are typically used for digital filter design."*  A documented boolean that has no effect
  cannot be intended.

## Suggested fix

Drop the parity factor from `_raised_cosine_window` and use
`n = window_length + periodic - 1`.  The upstream unit reference needs the same correction:
`tensorflow/python/kernel_tests/signal/window_ops_test.py::_scipy_raised_cosine` contains

```python
    odd = length % 2
    if not symmetric and not odd:
        length += 1
    ...
    if not symmetric and not odd:
        window = window[:-1]
```

i.e. it returns the *symmetric* window for odd lengths too, although its docstring says it "matches
SciPy" - which is why `test_hann_window` / `test_hamming_window` (parameterized over
`_WINDOW_LENGTHS = [1, 2, 3, 4, 5, 31, 64, 128]`, which includes odd lengths) pass despite the
bug.  With the helper corrected the existing tests reproduce the defect immediately; a regression
test should assert `hann_window(9, periodic=True)` equals
`scipy.signal.windows.hann(9, sym=False)` (or the closed form above) while `periodic=False` equals
`sym=True`.

## Controls (all in `repro.py`, 183 checks, 0 failing on each of 2 replays)

* **Parity control** - every odd `window_length` in `3..129` returns `periodic=True` bit-identical
  to `periodic=False`, and the error equals the symmetric-vs-periodic gap exactly.
* **Even-length control** - every even `window_length` in `2..128` differs between the two settings
  and matches its own reference to `1e-12`, so this is not a general window-function failure.
* **Reference control** - the comparison uses the closed form (equivalently
  `scipy.signal.windows.hann/hamming(..., sym=False)` and
  `torch.hann_window/hamming_window(..., periodic=True)`, checked separately), not another
  TensorFlow lane.
* **Lane control** - eager, `tf.function`, `tf.function(jit_compile=True)` and a **non-constant**
  `window_length` in graph mode all agree bitwise, so the defect is in the shared implementation and
  is not constant-folding or a compiler divergence.
* **`window_length == 1` control** - both settings return `[1.0]`, the documented special case.
* **Downstream control** - all-NaN-free, smooth results: the `tf.signal.stft` output matches the
  symmetric-window reference to `1e-9` and misses the periodic one by `4.25`.

## Novelty

Searched the TensorFlow issue tracker (issues and PRs) for `hann_window`, `hann_window periodic`,
`hamming_window` (title), `raised_cosine` and `window periodic odd` (body) on 2026-09-13: the only
hits are **#122530** (an open PR adding `blackman_window`), **#33897** (TFLite `Real` op) and
**#14158** (a 2017 Android `Cos` kernel issue) - nothing about periodicity or odd lengths, and
`raised_cosine` matches nothing.  The defect is still present at master, so it is unfixed.  Within
this campaign, the wave-191 `tf.signal` row compared `stft`/`inverse_stft` *across lanes*
(eager vs `tf.function` vs XLA) and found them consistent - which is exactly the reason this
parity-dependence went unnoticed: only a parity-aware reference exposes it.

## Environment

```
tensorflow 2.21.0 (CPU), numpy 2.5.1, Python 3.12.13, WSL2
references: scipy 1.18.0 (jax/jaxlib 0.10.2 interpreter), torch 2.12.1+cu130
```

## Anticipated objections and responses

* **"This is not a compiler bug."**  Correct - the implementation is wrong in every lane
  (eager/`tf.function`/XLA agree bitwise).  It is still a silent wrong-value defect in a public,
  documented API, found by differential testing against the reference definitions.
* **"A symmetric odd-length Hann window already has zero endpoints, so it is periodic."**  It is
  continuous under tiling, but the periodic/symmetric distinction in this API (and in scipy, PyTorch
  and MATLAB) is the denominator: a symmetric window samples the continuous taper at `N` points
  including both endpoints (`n/(N-1)`), a periodic one samples `n/N` and drops the closing sample.
  Returning one array for both settings makes the documented flag a silent no-op, which is the bug
  regardless of which convention one prefers.
* **"The unit test asserts the current behaviour, so it is intended."**  The test's own reference
  helper contains the same `if not symmetric and not odd` shortcut and claims to match SciPy; it
  does not (`scipy.signal.windows.hann(9, sym=False)` is the denominator-9 vector shown above).
* **"Odd frame lengths are not used in practice."**  They are legal, and `tf.signal.stft` accepts
  them; the failure is silent and, for small lengths (`3`, `5`, `9`), very large (`0.75`, `0.4`,
  `0.27`).
* **"Just pass `periodic=False`."**  That is what is happening already - the defect is that
  `periodic=True` cannot be requested for odd lengths at all.
