# `ops.stft(..., center=False)` rejects `n_fft == signal_length`, the documented single-frame STFT

**Component:** MindSpore 2.9.0 (CPU), `mindspore.ops.stft`, PyNative **and** GRAPH (`ms.jit`), CPU
kernel `STFT` (`Resize`, `stft_cpu_kernel.cc`).
**Severity:** a documented, well-defined configuration is rejected - the transform is *not computed*
and an internal kernel error is raised.  `ops.stft` cannot take the spectrum of a single
window-length segment (the most basic STFT use), cannot process signals with
`signal_length <= n_fft` when `center=False`, and also fails for a padded (`center=True`) input
whose padded length happens to equal `n_fft`, e.g. `len(x) = 1, n_fft = 3, pad_mode='CONSTANT'`.
PyTorch and SciPy return the expected single frame; MindSpore's own docstring defines that frame.
**Status:** present in 2.9.0 in PyNative and GRAPH mode, and still present at master
(`stft_cpu_kernel.cc:189`, `n_fft_ >= input_len_`).  There is no workaround short of padding the
signal by hand, which changes the requested transform.
**Found with:** the wave-370 Rondo STFT probe (the wave-369 window work led to the spectral API and
to a cross-framework grid of `(length, n_fft, hop, center, pad_mode)` against a hand-rolled NumPy
reference, PyTorch and SciPy); certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 38 checks).

---

## Summary

```python
import numpy as np, mindspore as ms
from mindspore import ops

x = ms.Tensor(np.arange(1.0, 9.0, dtype=np.float32))          # 8 samples
ops.stft(x, n_fft=8, hop_length=8, center=False, return_complex=True)
# RuntimeError:
#   [mindspore/ops/kernel/cpu/native/stft_cpu_kernel.cc:192] Resize]
#   For 'STFT', expected 0 < n_fft < 8, but got n_fft=8.
# the docstring defines   T = 1 + (len - n_fft) / hop_length = 1 + (8 - 8)/8 = 1 frame
# torch.stft(..., n_fft=8, hop_length=8, center=False) -> complex64 (5, 1)
```

`n_fft = signal_length - 1` with the same `hop_length` works (`(4, 1)`), `center=True` works, and the
accepted configurations return the reference values - only the equality is refused:

| `len(x)` | `n_fft` | `hop` | `center` | MindSpore 2.9.0 | documented `T` | PyTorch 2.12.1 |
|---|---|---|---|---|---|---|
| 8 | 8 | 8 | False | **RuntimeError** (kernel guard) | 1 | `(5, 1)` complex64 |
| 8 | 8 | 1, 2, 4 | False | **RuntimeError** | 1 | `(5, 1)` |
| 1 | 1 | 1 | False | **RuntimeError** | 1 | `(1, 1)` |
| 2 | 2 | 2 | False | **RuntimeError** | 1 | `(2, 1)` |
| 8 | 7 | 7 | False | `(4, 1)` | 1 | `(4, 1)` |
| 8 | 8 | 4 | True | `(5, 3)` | 3 | `(5, 3)` |
| 1 | 3 | 1 | True, `pad_mode='CONSTANT'` | **RuntimeError** | 1 | `(2, 1)` |

The failure is independent of `hop_length`, of `onesided` / `return_complex` / `normalized`, of
whether a `window` is supplied, of the dtype (fp32/fp64) and of the input rank (1-D or 2-D), and graph
mode behaves identically:

```
GRAPH n_fft=8, len=8 -> ERR RuntimeError      GRAPH n_fft=7, len=8 -> OK (4, 1)
```

SciPy agrees with PyTorch on the shape: `scipy.signal.stft(np.arange(1., 9.), nperseg=8, noverlap=0,
boundary=None, padded=False)` returns `Zxx` of shape `(5, 1)`.

## Root cause

`mindspore/ccsrc/plugin/device/cpu/kernel/stft_cpu_kernel.cc` (`STFTCpuKernelMod::Resize`), line 189
at master:

```cpp
  if (n_fft_ <= kSTFTNum0 || n_fft_ >= input_len_) {          // <- strict '>='
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', expected 0 < n_fft < " << input_len_
                  << ", but got n_fft=" << n_fft_ << ".";
    return KRET_RESIZE_FAILED;
  }
  ...
  n_frames_ = 1 + (input_len_ - n_fft_) / hop_length_;        // == 1 when n_fft_ == input_len_
```

The guard forbids exactly the value for which the very next line computes a valid, non-empty frame
count (`n_frames_ = 1`), and which the public docstring documents:

```
T = 1 + (len - n_fft) / hop_length,   len = signal_length when center is False
```

The Python-level validation is already inclusive - for `len = 4, n_fft = 8, center=False` it raises
`ValueError: For primitive[STFT], the n_fft must be in (0,4], but got 8.` - so `n_fft == len` passes
the front-end check and then fails inside the kernel with a `RuntimeError` carrying a C++ stack
instead of a user-facing message.  Relaxing the kernel guard to `n_fft_ > input_len_` makes the
documented case work; the window-padding and frame-count code paths already handle it.

## Impact

* **Single-frame spectral analysis is impossible.**  Computing the DFT of one window-length segment
  (a Hann-windowed frame, a spectral snapshot, a filter-bank front end) is the elementary STFT use;
  MindSpore requires the caller to pad the signal to `n_fft + 1` samples first, which is not the same
  transform and is not documented.
* **Short signals with `center=False`.**  Every `signal_length <= n_fft` fails: either the
  front-end `ValueError` (`n_fft > len`) or the kernel `RuntimeError` (`n_fft == len`).
* **Padded inputs can hit it too.**  With `center=True` the effective length is
  `signal_length + 2*(n_fft//2)`; for `signal_length = 1` and odd `n_fft` that equals `n_fft`, so
  `ops.stft([3.0], n_fft=3, center=True, pad_mode='CONSTANT')` also fails although PyTorch returns
  `(2, 1)`.
* The error is an internal kernel failure (`RuntimeError` + C++ stack), not one of the `TypeError`s
  the docstring lists, so it is also hard to diagnose.

## Suggested fix

Change the kernel guard to reject only `n_fft > input_len`:

```cpp
  if (n_fft_ <= kSTFTNum0 || n_fft_ > input_len_) {
```

and, if the front-end bound is meant to be `(0, input_len]`, keep the Python validation aligned with
it (it already is).  A regression test is the length-8 signal with `n_fft=8, hop_length=8,
center=False`, which must return shape `(5, 1)` (one-sided, complex) with the values of the
length-8 DFT of the frame.

## Controls (all in `repro.py`, 38 checks, 0 failing on each of 2 replays)

* **Equality control** - `n_fft = len - 1` is accepted for `len` = 2, 5, 9, 17, 64 and its frame
  count equals the documented formula, so the guard is about the equality and not about
  "n_fft close to the signal length".
* **Padding control** - `center=True` with the same `len`/`n_fft` is accepted and returns the
  documented frame count, so the defect is not a general "signal too short" rule.
* **Front-end control** - the Python check reports the *inclusive* bound
  (`n_fft must be in (0,4]` for `len = 4`), i.e. `n_fft == len` is meant to be legal; the kernel then
  raises `RuntimeError` for it.
* **Value control** - for the accepted configurations the output matches a hand-rolled NumPy STFT
  (framing + `np.fft.fft`, one-sided) to better than `1e-6` in fp32, and matches PyTorch to `~1e-6`
  on six configurations - the defect is the rejection, not the arithmetic.
* **Error-source control** - a clean child process must both fail and log
  `expected 0 < n_fft < 8, but got n_fft=8` on stderr, which ties the rejection to the kernel guard.
* **Lane control** - PyNative and `ms.jit` reject and accept the same configurations.

## Novelty

Searched the MindSpore tracker (GitHub mirror `mindspore-ai/mindspore`, issues and PRs) on
2026-09-13: `stft` returns **zero** hits and `n_fft` returns zero hits; the published docs for
`mindspore.ops.stft` / `mindspore.ops.STFT` document the `T = 1 + (len - n_fft)/hop_length` formula
and list no constraint beyond the `TypeError`s.  The kernel guard is still `n_fft_ >= input_len_` at
master, i.e. unfixed.  Campaign-internal: the only earlier `stft` coverage is the TensorFlow
`tf.signal` lane-consistency row (wave 191) and the wave-368/369 window rows - no MindSpore STFT
record exists in the ledger.

## Environment

```
mindspore 2.9.0 (CPU, PyNative and GRAPH), numpy 1.26.4, Python 3.12.13, WSL2
references recorded from: torch 2.12.1+cu130, scipy 1.18.0 (jax/jaxlib 0.10.2 interpreter) -
see collect_env.txt and evidence.txt
```

## Anticipated objections and responses

* **"`n_fft` must be smaller than the signal - that is a design rule."**  The public docstring's own
  frame-count formula gives `T = 1` for this case, the Python-level validation's bound is inclusive
  (`(0, len]`), and the line immediately after the guard computes `1 + (input_len - n_fft)/hop`.
  PyTorch and SciPy both return the frame.  Nothing documents the strict rule.
* **"Just use `center=True`."**  `center=True` pads the signal and changes the framing and output
  length (for `len = n_fft = 8, hop = 4` it returns 3 frames, not 1); it is not an equivalent call.
  And the padded-equality case (`len = 1, n_fft = 3, pad_mode='CONSTANT'`) still fails.
* **"This is only an error-message/validation issue."**  No output is produced for a valid request:
  the capability is missing, not just the message, and the message itself is an internal
  `RuntimeError` with a C++ stack.
* **"Pad the signal manually."**  That changes the signal and the transform (a zero-padded frame is
  not a single-length frame); it also cannot reproduce `center=False` semantics for short signals.
