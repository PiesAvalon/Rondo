# MindSpore window functions do not honour `dtype`: ignored for `window_length <= 1`, and `hamming_window` truncates `alpha`/`beta` to float32

**Component:** MindSpore 2.9.0 (CPU, PyNative and GRAPH/`ms.jit`), `mindspore.ops.hann_window`,
`mindspore.ops.hamming_window`, `mindspore.ops.kaiser_window` (and for contrast
`mindspore.ops.blackman_window`, `mindspore.ops.bartlett_window`).
**Severity:** silent contract violations.  (A) For `window_length <= 1` the `dtype` argument is
ignored completely - a Float16 or Float32 request returns a **Float64** tensor, and an *invalid*
dtype such as Int32 is silently accepted where the documented `TypeError` is raised for every other
length; the Float64 window then silently promotes float32 operands in `ms.jit` graphs.  (B)
`hamming_window(..., dtype=mindspore.float64)` carries only float32-precision coefficients: the
returned Float64 window is bit-for-bit the fp64 closed form built from `float32(0.54)` and
`float32(0.46)`, so it is wrong by up to 3.0e-8 for the default coefficients.
**Status:** present in 2.9.0 in PyNative and `ms.jit` (both return the same tensors), on CPU.  The
sibling windows (`blackman_window`, `bartlett_window`) and every length `>= 2` honour `dtype`
correctly; TensorFlow and PyTorch return the requested dtype for lengths 0 and 1, and PyTorch's
`hamming_window` is fp64-exact.
**Found with:** the wave-369 Rondo window probe (a cross-framework comparison of the same
`hann/hamming/kaiser` API after TF-028 exposed the parity dependence of the TensorFlow version);
certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`,
91 checks).

---

## Summary

```python
import mindspore as ms
from mindspore import ops

# (A) dtype is ignored for window_length <= 1
ops.hann_window(1, dtype=ms.float16).dtype     # Float64   (expected Float16)
ops.hann_window(0, dtype=ms.float32).dtype     # Float64   (expected Float32, empty)
ops.hamming_window(1, dtype=ms.float32).dtype  # Float64
ops.kaiser_window(1, dtype=ms.float16).dtype   # Float64
ops.hann_window(2, dtype=ms.float16).dtype     # Float16   (correct from length 2 on)
ops.blackman_window(ms.Tensor(1, ms.int32), dtype=ms.float16).dtype   # Float16 (sibling is correct)

# the early return bypasses the documented dtype validation as well
ops.hann_window(1, dtype=ms.int32)             # returns Float64 [1.0] instead of raising
ops.hann_window(2, dtype=ms.int32)             # TypeError: 'dtype' must be floating point dtypes

# (B) hamming_window's coefficients are float32 even for dtype=float64
ops.hamming_window(9, periodic=True, dtype=ms.float64).asnumpy()[0]
# 0.08000001311302185           <- float64(float32(0.54) - float32(0.46))
# exact 0.54 - 0.46 = 0.08000000000000002
# torch.hamming_window(9, periodic=True, dtype=torch.float64)[0] = 0.08000000000000002
```

### (A) `dtype` ignored for `window_length <= 1`

`hann_window`, `hamming_window` and `kaiser_window` return
`Tensor(np.ones(window_length))` **before** any dtype handling, and NumPy's `ones` is float64:

| call | returned dtype (Float16 requested) | (Float32 requested) | (Float64 requested) |
|---|---|---|---|
| `hann_window(0/1)` | **Float64** | **Float64** | Float64 |
| `hamming_window(0/1)` | **Float64** | **Float64** | Float64 |
| `kaiser_window(0/1)` | **Float64** | **Float64** | Float64 |
| `blackman_window(0/1)` (control) | Float16 | Float32 | Float64 |
| `bartlett_window(0/1)` (control) | Float16 | Float32 | Float64 |
| any of the five, length `>= 2` (control) | Float16 | Float32 | Float64 |

The values are correct (`[1.0]` for length 1, empty for length 0), so only the dtype metadata is
wrong - but the wrong metadata is what propagates:

```python
@ms.jit
def f(x):
    return x + ops.hann_window(1, dtype=ms.float64)

f(ms.Tensor(np.ones(3, np.float32))).dtype     # Float64  (a Float32 tensor silently becomes Float64)
ops.cast(x_f32, ops.hann_window(1, dtype=ms.float16).dtype)  # casts to Float64, not Float16
```

TensorFlow (`tf.signal.hann_window(1, dtype=tf.float16).dtype == float16`, and length 0 gives an
empty float16 tensor) and PyTorch (`torch.hann_window(1, dtype=torch.float16).dtype ==
torch.float16`) both honour the argument at the same lengths.

### (B) `hamming_window` truncates `alpha`/`beta` to float32

For every length and both periodic settings the returned Float64 tensor is **bit-for-bit equal** to
the fp64 closed form built from the float32-rounded coefficients, and differs from the exact fp64
window by up to `2.98e-08` (one float32 ulp of the coefficients):

| `N` | periodic | `max|window - exact fp64|` | window == fp32-coefficient reference | `window[0]` | exact |
|---|---|---|---|---|---|
| 2 | True | `2.98e-08` | **True** | `0.08000001311302185` | `0.08000000000000002` |
| 3 | True | `2.56e-08` | **True** | `0.08000001311302185` | `0.08000000000000002` |
| 5 | True | `2.82e-08` | **True** | `0.08000001311302185` | `0.08000000000000002` |
| 9 | True | `2.93e-08` | **True** | `0.08000001311302185` | `0.08000000000000002` |
| 17 | True | `2.97e-08` | **True** | `0.08000001311302185` | `0.08000000000000002` |
| 33 | True | `2.98e-08` | **True** | `0.08000001311302185` | `0.08000000000000002` |

(periodic=False rows behave the same; the full table is in `evidence.txt`.)

Two controls isolate the coefficients from the rest of the computation:

* `hamming_window(9, periodic=True, alpha=0.5, beta=0.5, dtype=ms.float64)` - coefficients that are
  exactly representable in float32 - matches the fp64 closed form **bit-for-bit**, so the cosine and
  the arithmetic are genuinely float64;
* `hann_window(9, dtype=ms.float64)`, `blackman_window(9, dtype=ms.float64)` and
  `bartlett_window(9, dtype=ms.float64)` are fp64-exact (bit-exact resp. within `1e-15`), so this is
  specific to `hamming_window`;
* arbitrary coefficients are truncated identically: `alpha=0.6, beta=0.4` gives
  `window[0] = 0.20000001788139343` where the exact value is `0.19999999999999996`
  (`torch.hamming_window(9, periodic=True, alpha=0.6, beta=0.4, dtype=torch.float64)[0]` is the
  exact one), and the whole vector equals the fp32-coefficient reference bit-for-bit.

## Root cause

```python
# mindspore/ops/function/math_func.py
6654: def hann_window(window_length, periodic=True, *, dtype=None):
...
6692:     if window_length <= 1:
6693:         return Tensor(np.ones(window_length))          # numpy ones are float64, dtype never applied
6697:     if dtype is not None and dtype not in mstype.float_type:   # validation happens *after*
6698:         raise TypeError(f"For 'hann_window', 'dtype' must be floating point dtypes, ...")

8422: def kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None)
8474:     if window_length <= 1:                              # same early return
8475:         return Tensor(np.ones(window_length))
# mindspore/ops/function/array_func.py
 413: def hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None)
 462:     if window_length <= 1:
 463:         return Tensor(np.ones(window_length))          # same
 468:     dtype = mstype.float32 if dtype is None else dtype
 469:     op = _get_cache_prim(P.HammingWindow)(periodic, alpha, beta, dtype)
```

For `hamming_window` the coefficients are float32 by op registration, independent of the requested
output dtype:

```python
# mindspore/ops/_op_impl/aicpu/hamming_window.py
21:    .attr("periodic", "bool")
22:    .attr("alpha", "float")        # <- float32 attribute
23:    .attr("beta", "float")         # <- float32 attribute
24:    .attr("dtype", "int")          # <- selects F16/F32/F64 output, cannot widen alpha/beta
```

so `.attr("dtype", ...)` selects the *output* type while `alpha`/`beta` are already float32; the
Python wrapper passes the Python floats straight into
`_get_cache_prim(P.HammingWindow)(periodic, alpha, beta, dtype)` (`array_func.py:469`).

## Impact

* **Type contract.** `ops.hann_window`/`hamming_window`/`kaiser_window` are documented to return the
  requested `dtype`.  In mixed-precision code (`float16` on Ascend/GPU, `float32` elsewhere) the
  length-1 window is unexpectedly `float64`: inside `ms.jit` it silently promotes the operand to
  Float64 (changing kernel selection and doubling memory traffic), and `ops.cast(x, w.dtype)` casts
  to the wrong type.  Requesting a non-floating dtype is documented to raise, and does for every
  length except 0/1, where it is silently accepted.
* **Precision.** A caller who asks for `dtype=mindspore.float64` specifically to avoid float32
  rounding still gets float32-rounded coefficients; the error (up to `3e-8`) is exactly the float32
  ulp of `0.54`/`0.46`, i.e. the returned Float64 tensor cannot represent the requested window.  Any
  fp64 reference comparison against `scipy.signal.windows.hamming` therefore fails at `~3e-8`.
  Wyckoff/Blackman-Harris style window design and high-dynamic-range spectral analysis are the
  typical fp64 users.
* Both defects are silent: no warning, no exception, plausible-looking values.

## Suggested fix

1. Move the dtype handling in front of the `window_length <= 1` early return (or cast the ones), e.g.

   ```python
   if dtype is not None and dtype not in mstype.float_type:
       raise TypeError(...)
   dtype = mstype.float32 if dtype is None else dtype
   if window_length <= 1:
       return Tensor(np.ones(window_length, dtype=np.float64), dtype)
   ```

   and do the same in `hamming_window` and `kaiser_window`.  (`blackman_window`/`bartlett_window`
   already behave correctly and can be used as the in-tree reference.)
2. Keep `alpha`/`beta` in the requested floating dtype: register the attributes as `double` (or pass
   them to the kernel as `float64`/template-typed values) so a `float64` `HammingWindow` uses fp64
   coefficients; the `dtype` attribute selects the output type and must not also narrow the inputs.

Regression tests: `hann_window(1, dtype=ms.float16).dtype == ms.float16`, the same for
`window_length = 0`, `hamming_window(1, dtype=ms.int32)` raising `TypeError`, and
`hamming_window(9, dtype=ms.float64)[0] == 0.54 - 0.46` exactly.

## Controls (all in `repro.py`, 91 checks, 0 failing on each of 2 replays)

* **Sibling control** - `blackman_window` and `bartlett_window` honour `dtype` at lengths 0 and 1
  (they take a Tensor length and go through the primitive's dtype handling), which rules out "the
  API cannot express this".
* **Length control** - all five ops honour `dtype` for `window_length >= 2`, so the failure is the
  `<= 1` path and not a general dtype plumbing problem.
* **Validation control** - `hann/hamming/kaiser_window(2, dtype=ms.int32)` raise the documented
  `TypeError` while the length-1 call returns a Float64 tensor, showing the early return also skips
  validation.
* **Value control** - the length-1 values are exactly `[1.0]` and the length-0 values empty, so the
  defect is metadata only (and the graph-mode promotion is measured on real values `[2.0, 2.0, 2.0]`).
* **Coefficient control** - `alpha=beta=0.5` (exactly representable in float32) is fp64-exact
  bit-for-bit while `0.54/0.46` is not, which isolates the truncation to the coefficients rather
  than the kernel arithmetic.
* **Cross-op control** - `hann_window`, `blackman_window` and `bartlett_window` are fp64-exact for
  the same length and dtype, so this is not a global "window functions are fp32" statement.
* **Lane control** - PyNative and `ms.jit` return bit-identical tensors for both defects, so this is
  the shared implementation and not a graph-mode lowering issue.

## Novelty

Searched the MindSpore tracker (GitHub mirror `mindspore-ai/mindspore` issues and PRs, plus the
published docs for `mindspore.ops.hann_window`, `hamming_window`, `kaiser_window`, `HammingWindow`
and `BlackmanWindow`) on 2026-09-13 for window dtype/length-1/alpha/beta/float64 combinations: no
matching report - the only hits are unrelated (the word "Windows" in titles, and the surrounding
`mindspore.mint.mean` empty-tensor family reported in 2026 by another user).  The campaign's own
nearest drafts are MS-035 (`ops.matrix_power` float32 dispatch, a different op and a different
mechanism: output values, not coefficients), MS-011/MS-004/MS-007/MS-008 (float32 kernels for fp64
activations/reductions), MS-025 (empty-tensor `cumsum`), all different operators; nothing covers the
`dtype <= 1` early return or the `HammingWindow` float attributes.

## Environment

```
mindspore 2.9.0 (CPU, PyNative and GRAPH), numpy 1.26.4, Python 3.12.13, WSL2
references recorded from: torch 2.12.1+cu130, tensorflow 2.21.0 (CPU), scipy 1.18.0 - see
collect_env.txt and evidence.txt
```

## Anticipated objections and responses

* **"`window_length <= 1` is a degenerate case."**  It is a documented, accepted input (the
  docstrings show the values), the sibling functions handle it correctly, and the consequence is a
  silent dtype change inside compiled graphs - the cheapest possible fix is one line.
* **"Float32 coefficients are fine for a window function."**  Then the coefficient parameters
  should be documented as float32, and `dtype=float64` should not be advertised.  `torch.hamming_window`
  with the same signature is fp64-exact, and the other four MindSpore windows are fp64-exact, so the
  asymmetry is a defect rather than a convention.
* **"This is the same as the known fp32-kernel family."**  No: the output dtype here *is* Float64 and
  the arithmetic is genuinely fp64 (`alpha=beta=0.5` is bit-exact) - only the two coefficients are
  narrowed, and the op registration shows exactly why.  The length-1 defect is a Python early-return
  path, not a kernel at all.
* **"`kaiser_window` at very large `beta` is also broken."**  True but **not claimed here**: for
  `beta >= 710` the implementation's `np.i0(beta)` overflows to `inf` and the window becomes
  zero/NaN, which `scipy.signal.windows.kaiser` reproduces exactly (same warning, same values), so it
  is a shared reference limitation rather than a MindSpore defect.  It is recorded as a boundary in
  `results/w369_ms_windows.py` and `KNOWN-BOUNDARIES.md`.
