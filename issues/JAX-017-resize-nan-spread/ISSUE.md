# `jax.image.resize` propagates a single NaN input to **every** output (zero weights in the dense contraction)

**Component:** JAX 0.10.2 (`jaxlib` 0.10.2), `jax.image.resize` and `jax.image.scale_and_translate`,
CPU and CUDA, `float32` and `float64`.
**Severity:** silent wrong values.  One NaN pixel in the input makes the *whole* resized image NaN
for every method except `nearest` (`linear`, `cubic`, `lanczos3`, `lanczos5`), even though the
interpolation kernel provably does not read that pixel for most outputs: in the minimal witness the
NaN input pixel has weight **exactly `0.0`** for 10 of the 13 outputs, and those 10 outputs come back
`NaN` instead of finite.  torch, SciPy and a local reference all keep the NaN local.
**Status:** present in 0.10.2 on CPU and CUDA, fp32 and fp64, eager, `jit` and 1-D/2-D; `nearest`
(a gather) is unaffected, which isolates the cause to the dense weight-matrix contraction.
**Found with:** the wave-347 Rondo sweep of the `jax.image` resampling surface (identity, constant
preservation, NumPy bilinear/nearest references, antialias semantics, NaN/edge inputs, CPU-vs-GPU);
certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`,
33 checks).

---

## Summary

```python
import jax, jax.numpy as jnp
from jax import image as jim
x = jnp.asarray([1., 2., 3., float("nan"), 5., 6., 7.])

jim.resize(x, (13,), method="linear")
# Array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], dtype=float64)
#        ^ input pixel 3 has weight exactly 0.0 for outputs 0-4 and 8-12,
#          so those ten outputs are provably independent of the NaN
```

The weight matrix that JAX builds internally (`jax._src.image.scale.compute_weight_mat`) for
`7 -> 13`, triangular kernel, is:

```
row for input pixel 3 (the NaN):
  [0.     0.     0.     0.     0.     0.4615 1.     0.4615 0.     0.     0.     0.     0.    ]
exactly-zero weights: outputs [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]   (10 of 13)
nonzero support:      outputs [5, 6, 7]
```

## Root cause

`jax/_src/image/scale.py`, `_scale_and_translate`, builds a **dense** `(input_size, output_size)`
weight matrix per spatial dimension and contracts it with the whole input:

```python
w = compute_weight_mat(m, n, scale[i], translation[i], kernel, antialias, ...)
contractions.append(w)
...
return jnp_einsum.einsum(x, in_indices, *contractions, precision=precision)
```

Every output sums over **all** input pixels, including those with weight exactly `0.0`, and
`0.0 * NaN == NaN` in IEEE arithmetic, so the NaN survives the sum for every output.  A dense
`np.einsum` on the same weight matrix reproduces it exactly (13/13 NaN); masking the *product*
instead of the weight (`where(w != 0, x*w, 0)`) restores the correct 3-output footprint.  The
`nearest` path is a gather (`_resize_nearest`) and therefore local.

## Impact

* Any resampling pipeline that can see a single invalid sample (masked/padded feature maps, invalid
  depth or flow, a NaN produced upstream) turns the entire resized image into NaN - silently, on
  both backends and both widths - where torch/SciPy produce a finite image with a small NaN
  neighbourhood.  Downstream losses, masks and `isfinite` checks all break at once.
* The failure is invisible in ordinary tests: with finite inputs JAX matches the local NumPy
  reference to fp64 rounding (max `8.9e-16`), so only the NaN footprint differs.

## Suggested fix

Do not multiply the NaN by zero weights: mask the product, not the weight, when building the
contraction (or restrict each output's contraction to the nonzero support of its kernel, e.g. a
gather/scatter over the taps):

```python
# instead of  sum_i w[i, o] * x[i]
contrib = jnp.where(w != 0, x[..., None] * w, 0.0)
out = contrib.sum(axis=input_axis)
```

Regression test: `resize([1..7] with x[3]=nan, (13,), method='linear')` must be NaN at exactly
`{5, 6, 7}` and finite elsewhere (and the same for 2-D, cubic/lanczos, both antialias settings).

## Controls (all in `repro.py`, 33 checks, 0 failing on each of 2 replays)

* **Weight-matrix proof** — `compute_weight_mat(7, 13, 13/7, 0, triangle, antialias=False)` has
  exactly 10 zeros in the row for the NaN pixel; `where(w != 0, x*w, 0)` gives NaN only at `[5, 6, 7]`.
* **`nearest` control** — the gather path returns exactly 1 NaN of 13 (and 1 of 25 in 2-D) and the
  exact nearest-neighbour values, so the defect is in the dense contraction, not in the input.
* **Geometry control** — with a finite input, `resize(..., method='linear')` matches the local
  reference to fp64 rounding (max `8.9e-16`), so the mapping/weights themselves are correct.
* **Cross-framework controls** — 2-D `6x7 -> 13x13` with one NaN: JAX `169/169` NaN; the local NumPy
  reference `15/169`; `torch.nn.functional.interpolate` (bilinear, `align_corners=False`) `15/169`
  (identical to the reference); `scipy.ndimage.zoom(order=1)` `20/169`.
* **Scope** — every non-`nearest` method, both `antialias` settings, fp32/fp64, eager/`jit`, CPU/CUDA,
  1-D and 2-D, and `scale_and_translate` (which shares `_scale_and_translate`) all reproduce;
  `scale_and_translate(method='nearest')` raises the documented `ValueError`.

## Novelty

Searched the JAX tracker for `image.resize` + NaN, resize NaN propagation, `scale_and_translate` NaN
and resize einsum zero weights (all zero hits).  Adjacent items found and distinguished:

* **jax-ml/jax#6780** (issue, closed) / **#6781** (PR, closed, 2021) "nans in gradient of
  `scale_and_translate`" — the *backward pass* producing NaNs from the weight normalisation, fixed in
  2021; this report is the forward NaN footprint with a NaN input, which that fix does not cover.
* **jax-ml/jax#37925** (issue, open) — `image.resize` `JaxRuntimeError: INTERNAL` for lanczos3/lanczos5
  on CUDA (autotuner), a compilation failure, not a value.
* **jax-ml/jax#8452** (open) — "Make image resizing primitive", a feature request.
* **jax-ml/jax#7591** (closed) — zero-size image resizing, a shape error.

No item describes a NaN input spreading to outputs whose weight is exactly zero.

## Environment

```
jax 0.10.2 / jaxlib 0.10.2, numpy 2.5.3, scipy (see collect_env.txt)
Python 3.12.13, WSL2, CPU and CUDA (RTX 4090)
```

## Anticipated objections and responses

* **"NaN in the input is invalid usage."**  A NaN is a legitimate float; the question is the NaN
  footprint, and JAX's own `nearest` path, torch, SciPy and the local reference all keep it local.
  The outputs that become NaN are those with weight exactly `0.0` from the NaN pixel - the kernel
  provably does not use it.
* **"`0 * NaN = NaN` is IEEE, this is expected."**  It is IEEE for a naive dense contraction, which is
  precisely the implementation choice being reported; materialising zero-weight terms is a bug in a
  resampling routine whose contract is a local filter, and it is trivially avoidable by masking the
  product.
* **"Just use `nearest`."**  The affected methods are the default `linear` and the high-quality
  `cubic`/`lanczos*` used for downsampling; `nearest` is not a substitute.
* **"The antialias/downsampling kernel can legitimately reach the NaN."**  True when the kernel is
  wider than the image, but the witness is *upsampling* `7 -> 13`, where the triangular support is 3
  outputs wide and the 10 other weights are exactly zero.
