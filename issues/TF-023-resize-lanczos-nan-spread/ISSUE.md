# Eager `tf.image.resize` spreads a single NaN to (almost) every output for the `scale_and_translate` methods

**Component:** TensorFlow 2.21.0 (CPU), eager and `tf.function`: `tf.image.resize` with
`method='lanczos3'`, `'lanczos5'`, `'gaussian'`, `'mitchellcubic'`, and with `antialias=True` for
`'bilinear'`/`'bicubic'`; `float32` and `float64`.
**Severity:** silent wrong values.  One NaN pixel makes 156/169 (`lanczos3`) or **169/169**
(`lanczos5`) of the resized outputs NaN, including outputs whose interpolation kernel provably cannot
reach the NaN pixel; the other resize methods in the same function stay local (4–77 of 169).
**Status:** present in 2.21.0 in **eager** mode (no `tf.function` and no XLA involved), so it is
independent of the XLA NaN-propagation reports filed against `tf.image.resize`: those describe
`jit_compile=True` over-propagating where eager is correct, whereas here eager itself over-propagates
for these methods.
**Found with:** the wave-351 Rondo sweep of TensorFlow's resampling surface (a JAX-017 follow-up); the
certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`,
42 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf
z = np.arange(1.0, 29.0).reshape(1, 4, 7, 1)     # 4x7, upsampling to 13x13 (scale 3.25 x 1.86)
z[0, 2, 3, 0] = float("nan")                     # one NaN pixel

tf.image.resize(z, (13, 13), method="lanczos3").numpy()   # 156 of 169 outputs NaN
tf.image.resize(z, (13, 13), method="lanczos5").numpy()   # 169 of 169 outputs NaN
```

The NaN footprint by method (same input, `4x7 -> 13x13`):

| method | NaN outputs | corner (0,0)? |
|---|---|---|
| `nearest` | 4/169 | finite |
| `bilinear` (`antialias=False`) | 18/169 | finite |
| `bicubic` (`antialias=False`) | 77/169 | finite |
| `area` | 12/169 | finite |
| `gaussian` | 117/169 | **NaN** |
| `mitchellcubic` | 130/169 | **NaN** |
| `lanczos3` | 156/169 | **NaN** |
| `lanczos5` | 169/169 | **NaN** |
| `bilinear` `antialias=True` | 55/169 | **NaN** |
| `bicubic` `antialias=True` | 130/169 | **NaN** |

## The output corner cannot read the NaN pixel

Upsampling `4x7 -> 13x13` with output index 0 maps to the source coordinate
`s = (0 + 0.5)/scale - 0.5` (TF's `scale_and_translate` convention), and a radius-`r` kernel spans
`± r/scale` in input-index units:

| kernel | row support of output 0 | col support of output 0 | reaches input (2,3)? |
|---|---|---|---|
| lanczos3 (`r=3`) | `[-1.269, 0.577]` -> rows 0 | `[-1.846, 1.385]` -> cols 0–1 | no |
| lanczos5 (`r=5`) | `[-1.885, 1.192]` -> rows 0–1 | `[-2.923, 2.462]` -> cols 0–2 | no |

So output `(0, 0)` is provably independent of the NaN at input `(2, 3)` (rows and columns both
excluded), yet it comes back NaN for `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic` and the
`antialias=True` variants.

## Root cause

`tf.image.resize` dispatches per method in
`tensorflow/python/ops/image_ops_impl.py::resize_images_v2`:

```python
scale_and_translate_methods = [ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5,
                               ResizeMethod.GAUSSIAN, ResizeMethod.MITCHELLCUBIC]
...
if method == ResizeMethod.BILINEAR:
    if antialias: return resize_with_scale_and_translate('triangle')
    else:         return gen_image_ops.resize_bilinear(images_t, new_size, half_pixel_centers=True)
...
elif method == ResizeMethod.BICUBIC:
    if antialias: return resize_with_scale_and_translate('keyscubic')
    else:         return gen_image_ops.resize_bicubic(images_t, new_size, half_pixel_centers=True)
elif method in scale_and_translate_methods:
    return resize_with_scale_and_translate(method)
```

The gather-based kernels (`resize_bilinear`, `resize_bicubic`, `resize_nearest_neighbor`,
`resize_area`) keep the NaN local.  The `scale_and_translate` path builds **dense** per-dimension
kernel matrices over the whole input extent and contracts them, so every output sums terms with
weight exactly `0.0`; `0.0 * NaN == NaN` survives the sum and poisons the output.  The raw op
reproduces the `tf.image.resize` footprints exactly (156/169 and 169/169), which pins the mechanism.

The `antialias=True` case is the cleanest demonstration that this is a masking artefact rather than a
wider kernel: on **upsampling** the antialias filter is documented to leave the result unchanged, the
finite values differ by `< 2e-6` (`bilinear`), yet the NaN footprint jumps from 18/169 to 55/169
purely because the `scale_and_translate` kernel carries additional zero-weight taps.

## Impact

* Any resize whose method is `lanczos3`/`lanczos5`/`gaussian`/`mitchellcubic` - or `bilinear`/`bicubic`
  with `antialias=True`, the recommended setting for downsampling - turns a whole image NaN from one
  invalid sample (masked/padded features, invalid depth/flow, an upstream NaN), silently, in eager
  mode, while the same call with `antialias=False` stays local.
* The constant-preservation invariant is unaffected (a constant image survives to `7e-7`), so the
  failure appears only with non-finite input and is invisible in ordinary accuracy tests.

## Suggested fix

Mask the product rather than the weight when the request is off the edge / outside the kernel
support, or restrict each output's contraction to the nonzero support of its kernel (the gather used
by `resize_bilinear` already does this).  A targeted change: in the `scale_and_translate` kernel
implementation, compute `where(w != 0, x * w, 0)` instead of multiplying the input by a zero weight.

Regression test: `tf.image.resize` of a `4x7` tensor with one NaN at `(2,3)` to `13x13` must be NaN at
exactly the outputs whose kernel support includes `(2,3)`, for every method and for
`antialias=True/False` (and the far corner must stay finite).

## Controls (all in `repro.py`, 42 checks, 0 failing on each of 2 replays)

* **Provable support bound** — the output `(0,0)` intervals above are computed in the repro and
  asserted to exclude `(2,3)` for `lanczos3` and `lanczos5`, yet both return NaN there.
* **Local-method controls** — `nearest`, `bilinear` (`antialias=False`), `bicubic`
  (`antialias=False`) and `area` keep the corner finite and the footprint small (4/18/77/12 of 169);
  `torch.nn.functional.interpolate` gives the same local footprints (6/18/77/12).
* **Value control** — on upsampling, `bilinear` with `antialias=True` vs `False` differs by `< 2e-6`
  in value while the NaN footprint grows from 18 to 55, isolating the defect to the zero-weight terms.
* **Mechanism control** — `tf.raw_ops.ScaleAndTranslate` with the same geometry and `kernel_type`
  reproduces the footprints exactly (156/169, 169/169) and the corner NaN.
* **Invariant controls** — a constant image stays constant to `< 1e-6` and an identity resize is
  within `1e-6` on the same dense path, so the weights themselves are fine.
* **Lane control** — eager and `tf.function` agree; no XLA/`jit_compile` is used anywhere in the repro.

## Novelty

The tracker has a family of `tf.image.resize` NaN reports, all of which describe **XLA over-propagating
where eager is correct**, and all of them for **bilinear** only:

* **tensorflow#119357** (open, `comp:ops` + `comp:xla`, stale) "Global NaN propagation bug in
  `tf.image.resize` (bilinear)" - the reproducer compares `eager_sum` against
  `tf.function(jit_compile=True)` and asserts they match; the bug is the XLA lane.
* **tensorflow#118382** (open) and **#117803** (closed, fixed) "XLA bridge over-propagates NaN to
  entire channel" - explicitly "Eager: 9 // XLA: 16", i.e. eager is the reference.
* **tensorflow#120240** (closed) is the same XLA bilinear report.

Searches for `ScaleAndTranslate` + NaN, `lanczos` + NaN, `image.resize` + NaN propagation return none
of these for the eager dense path.  This report differs in lane (eager is wrong), in the dispatch
branch (`gen_image_ops.scale_and_translate`, not the XLA bridge) and in the affected methods
(`lanczos3/5`, `gaussian`, `mitchellcubic`, `antialias=True`).  Fixing the XLA reports by matching XLA
to eager would leave this behaviour untouched, and the `antialias=True` control (values unchanged,
footprint tripled) is not part of any of them.

## Environment

```
tensorflow 2.21.0 (CPU), numpy 2.5.3, Python 3.12.13, WSL2 (see collect_env.txt)
```

## Anticipated objections and responses

* **"This is the known XLA NaN issue."**  No XLA is involved: `repro.py` runs in eager mode and
  `tf.function` (non-XLA) gives the same result.  The filed issues have eager as the *correct*
  reference lane and only cover `bilinear`.
* **"A NaN pixel is an invalid input."**  A NaN is a legitimate float value, and the question is the
  footprint; the same function's other methods, and `torch`, keep it local.  The corner output is
  provably independent of the NaN pixel.
* **"The kernel legitimately widens when `antialias=True`."**  Only when downsampling; the witness
  upsamples, where the finite values are unchanged to `< 2e-6` yet the NaN footprint triples.
* **"`0 * NaN = NaN` is just IEEE."**  It is IEEE for a dense contraction, which is the implementation
  being reported; the gather branches in the same op avoid it, and masking the product fixes it.
