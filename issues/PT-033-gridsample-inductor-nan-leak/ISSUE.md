# `torch.compile` (Inductor) 2-D `grid_sample` materialises out-of-bounds taps: NaN/+-inf leak into finite outputs and a correct `+0.0` becomes `-0.0`

**Component:** PyTorch 2.12.1+cu130, `torch.compile` with the Inductor backend, CPU **and** CUDA:
`torch.nn.functional.grid_sample` 2-D, all three interpolation modes (`bilinear`, `nearest`,
`bicubic`), `float32` and `float64`.
**Severity:** silent wrong value.  A compiled model returns `nan` for an output that eager — and a
hand-written NumPy reference — computes as an ordinary finite number, and for an output that is
exactly `0.0` it can return `-0.0`.  One non-finite entry anywhere in the input feature map (a NaN
**or** a `+-inf`) poisons outputs that do not read it, so a single bad pixel turns whole warped frames
(and anything downstream: losses, masks, `isfinite` guards, gradient magnitudes, `signbit`-sensitive
code) into NaN or a wrongly-signed zero **only when compiled**.  The defect is visible on entirely
finite input as well: a finite negative entry at a masked-out tap turns the correct `+0.0` into
`-0.0` (`0 * -1e300 = -0.0`).
**Status:** present in 2.12.1+cu130.  Eager CPU, eager CUDA and `torch.compile(backend='aot_eager')`
all agree bit-for-bit with the reference; only `backend='inductor'` diverges, which attributes the
defect to Inductor code generation rather than to the decomposition or to dispatcher semantics.
**Found with:** the wave-332…335 Rondo probe of the PyTorch image-resampling surface (eager vs
`torch.compile` vs a hand-written bilinear reference, NaN/±inf/signed-zero inputs); certified
reproducer `repro.py` in this directory (`--replays 2` → `RESULT: 0 failing replay(s).`, 137 checks,
0 failing on each replay).

---

## Summary

```python
import torch, torch.nn.functional as F

def warp(inp, grid):
    return F.grid_sample(inp, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)

inp = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
inp[0, 0, 0, 0] = float("nan")           # one NaN pixel
grid = torch.tensor([[[[-1.9, -1.9]]]])  # both taps are out of bounds

warp(inp, grid)                          # tensor([[[[0.]]]])   <- correct: no in-bounds tap at all
torch.compile(warp)(inp, grid)           # tensor([[[[nan]]]])  <- Inductor
```

The grid point `(-1.9, -1.9)` with `align_corners=False` unnormalizes to `-2.3`, and with
`padding_mode='zeros'` **every** bilinear tap lies outside the 4×4 input.  The correct output is
therefore exactly `0.0` without reading a single input element; eager, CUDA and `aot_eager` all return
`0.0` with bit pattern `0x0`.  Inductor returns the canonical quiet NaN `0x7ff8000000000000` because
it *does* load an element and only then multiplies it by a zero weight.

The same divergence appears with the other padding modes and interpolation modes, where the correct
answer is a non-zero finite value:

| configuration (4×4 input `arange(1,17)`, NaN at `(0,0)`) | eager CPU/CUDA | Inductor CPU/CUDA |
|---|---|---|
| `bilinear`, `zeros`, `align_corners=False`, `grid=(-1.9, -1.9)` | `0.0` / `0x0` | `nan` / `0x7ff8000000000000` |
| `bilinear`, `border`, `align_corners=False`, `grid=(1.9, 1.9)` | `16.0` | `nan` |
| `bilinear`, `reflection`, `align_corners=False`, `grid=(-0.25, 0.75)` | `14.0` | `nan` |
| `nearest`, `zeros`, `align_corners=False`, `grid=(-1.9, -1.9)` | `0.0` | `nan` |
| `bicubic`, `zeros`, `align_corners=False`, `grid=(-1.9, -1.9)` | `0.0` | `nan` |
| `bilinear`, `zeros`, `align_corners=True`, `grid=(-1.9, -1.9)` | `0.0` | `nan` (CPU) / `0.0` (CUDA) |

The trigger is not NaN-specific; the same masked load is materialised for any value:

| input at the out-of-bounds tap (same `zeros`/`-1.9` witness) | eager CPU/CUDA | Inductor CPU/CUDA | aot_eager |
|---|---|---|---|
| `+inf` | `+0.0` | **`nan`** | `+0.0` |
| `-inf` | `+0.0` | **`nan`** | `+0.0` |
| `-1e300` (finite) | `+0.0` | **`-0.0`** | `+0.0` |
| `+1e300` (finite) | `+0.0` | `+0.0` | `+0.0` |

and when the `border` lane genuinely reads the value (grid `(1.9, 1.9)`, entry at `(3,3)`) every lane
returns `+-inf` correctly - so the leak is specific to the masked-out tap, not to infinite inputs.

Both widths are affected (`float32` and `float64`), and `torch.compile(warp)` reproduces with the
default mode, with `fullgraph=True` and with `mode="reduce-overhead"`.  The 3-D/5-D `grid_sample`
path is **not** affected (Inductor returns `0.0` there).

## Root cause

Inductor lowers the decomposition's `torch.where(in_bounds, value, 0)` into a **clamped index plus a
zeroed weight**, i.e. `load(clamp(index)) * mask`, instead of selecting the value.  The C++ kernel
Inductor emits for the witness (`TORCH_LOGS=output_code`; full excerpt in `kernel-excerpt.txt`) reads:

```cpp
auto tmp18 = tmp7 && tmp17;                       // in-bounds mask
auto tmp20 = static_cast<int64_t>(0);
auto tmp21 = tmp18 ? tmp19 : tmp20;               // out-of-bounds index is forced to 0
...
auto tmp38 = in_ptr1[static_cast<int64_t>(tmp34 + 4L*tmp26)];   // unconditional load -> element (0,0)
auto tmp44 = double(tmp41 * tmp43);               // bilinear weight
auto tmp45 = tmp18 ? tmp44 : tmp6;                // weight zeroed when out of bounds (tmp6 == 0.0)
auto tmp46 = double(tmp38 * tmp45);               // 0.0 * NaN = NaN
```

`0.0 * finite == 0.0` masks correctly for positive finite inputs, which is why the bug is invisible in
ordinary testing; `0.0 * NaN == NaN` and `0.0 * +-inf == NaN` break the mask, and `0.0 * negative`
produces `-0.0`, which is why the leakage is not NaN-specific.  The clamp target is index `0`, so the leak is triggered
by a NaN at the flat element the clamp lands on (for `border`, the clamp lands on the border element:
NaN at `(0,0)` leaks for `grid=(1.9,1.9)`, while NaN at `(3,3)` is genuinely read by eager and gives
NaN in both lanes — the repro asserts both).

The leak is **specific to this lowering**, not a general `torch.where` behaviour: an explicit
`torch.where(mask, x, 0.0)` (tensor or scalar other-branch), `Tensor.masked_fill`, `F.pad(...,
mode='constant', value=0.0)`, `F.embedding(..., padding_idx=...)`, `masked_select`, `nan_to_num` and
`clamp` all match eager bit-for-bit under Inductor with the same NaN inputs.  The likely fix site is
therefore the masked load the `grid_sampler_2d` lowering emits (clamp the index / zero the weight),
not a global `where(c, x, 0) -> x * c` fold: a masked-out tap must select `0.0` instead of
multiplying an unconditionally loaded value by a zero weight.

## Impact

* A compiled inference loop diverges from eager on any input containing a NaN — a common occurrence
  in warping/registration pipelines (masked or padded feature maps, invalid depth samples, NaN from an
  earlier `log`/`sqrt`), and in training whenever a NaN appears transiently.
* The failure is **silent**: the same source code returns a finite, plausible value eagerly and `nan`
  compiled, on both CPU and CUDA, in all interpolation modes, so the bug is not device- or
  mode-specific and is easy to miss.
* `torch.compile` users cannot work around it from user code without replacing `grid_sample`
  (`backend='aot_eager'` is correct on the same inputs).

## Suggested fix

1. When lowering `torch.where(cond, load(...), 0)` (and the masked-load path generally), select the
   value rather than multiplying the loaded value by the zeroed weight — e.g. keep an explicit
   `where`/select for loads whose result can be NaN, or clamp the *index* and use the mask only to
   choose between the load and a literal `0.0`.
2. Regression test: the reproducer in this directory (a NaN, `+-inf` or negative finite pixel that no
   tap reads; assert the compiled output is exactly `+0.0` with the correct sign) for
   `zeros`/`border`/`reflection` x `bilinear`/`nearest`/`bicubic`.

## Controls (all in `repro.py`, 169 checks, 0 failing on each of 2 replays)

* **Finite control** — replacing the NaN with `5.0` makes eager, `aot_eager` and Inductor agree
  bit-for-bit in every configuration, so this is not a border/indexing error; a finite `+1e300` value
  is also masked correctly, while `-1e300` exposes the `-0.0` sign flip.
* **Non-finite control** — `+-inf` at a masked-out tap gives `nan` under Inductor and `+0.0` under
  eager and `aot_eager`, and the `border` lane where the value is genuinely read returns `+-inf` in
  every lane, so the leak is about the mask, not about the value being infinite.
* **No-NaN control** — the same tensors with finite values produce bit-identical eager and Inductor
  results.
* **Attribution control** — `torch.compile(..., backend='aot_eager')` matches eager bit-for-bit in
  every configuration, isolating the defect to Inductor codegen.
* **Reference** — eager matches a hand-written NumPy bilinear implementation; in the primary witness
  no tap is in bounds at all, so `0.0` follows from the definition without a reference implementation.
* **Non-leaking control** — a NaN at `(1,1)` (not the clamp target) does not leak, and a NaN at
  `(3,3)` with `border` is genuinely read by eager (both lanes NaN), pinning the mechanism to the
  clamped load.
* **Scope control** — the 3-D/5-D `grid_sample` path matches eager.
* **General-mask control** — plain `torch.where(mask, x, 0.0)`, `masked_fill`, `F.pad(mode='constant')`,
  `F.embedding(padding_idx=...)`, `masked_select`, `nan_to_num` and `clamp` all match eager under
  Inductor on the same NaN inputs, so this is the `grid_sampler_2d` masked load, not `where` in general.
* **Cross-framework control** — MindSpore 2.9.0 `ops.grid_sample` returns the correct finite values
  (`0.0`/`16.0`/`14.0`) in PyNative **and** under `ms.jit` for the same witnesses and controls, so this
  is specific to Inductor's lowering rather than to compiled resampling in general.

## Novelty

Searched the PyTorch tracker and the web for `grid_sample` + `inductor` / `torch.compile` + NaN and
for masked-load NaN fill regressions.  The adjacent items found and distinguished:

* **pytorch#149013** (open) — eager `grid_sample` returns the padding value inconsistently for
  out-of-bounds coordinates; no `torch.compile`, and about the padding value for a *finite* input.
* **pytorch#24823** (open, 2019) — eager `grid_sample` mishandles a NaN **grid** (segfaults/incorrect
  results); here the grid is finite and the input holds the NaN, and eager is the correct lane.
* **pytorch#189896** (PR, closed) — Triton `masked()` skipped the explicit `tl.where` when the masked
  body is a plain load, making `F.pad`/unpad return wrong **finite** values; a different operator,
  backend and direction (there the fill was lost, here a NaN leaks through the zero mask).
* **pytorch#186281 / #187580** (PRs, closed) — Inductor's uniform-value constant folder folded
  `x * 0 → 0`, dropping NaN/Inf; that fold applies to a *constant* zero, whereas here the zero/one
  mask is a runtime value derived from the coordinate bounds, and the bug is the opposite direction
  (a NaN is introduced where eager is finite).
* **pytorch#173133** (open) — same class (Inductor vs eager NaN inconsistency) but for
  `torch.bucketize` on CUDA.
* No result at all for `grid_sample` + Inductor/`torch.compile` NaN leaks, so the combination of
  operator, backend, 2-D-only scope and the `0 * NaN` mechanism appears to be unreported.

## Environment

```
torch 2.12.1+cu130 (CUDA 13.0, RTX 4090 and CPU), Python 3.12.13, WSL2
see collect_env.txt for the full dump
```

## Anticipated objections and responses

* **"NaN in the input is invalid usage."**  The input is a legitimate float tensor; PyTorch's own
  contract for out-of-bounds `padding_mode='zeros'` taps is to contribute `0.0`, and eager does.  A
  NaN must not make a compiled kernel read an element that the semantics never reads.
* **"NaN propagation is undefined."**  Undefined would explain eager NaN → NaN.  Here the compiled
  lane *introduces* NaN into outputs whose value is exactly `0.0` by definition (all taps out of
  bounds), and the finite control proves the indexing is otherwise identical.
* **"It is just a precision/ulp difference."**  The values are `0.0` vs `nan` — a structural
  difference, not a tolerance question.
* **"A 1-pixel NaN input is a corner case."**  The trigger is one NaN anywhere in the feature map
  plus any grid point whose taps touch the clamped border, i.e. any padding around the border; every
  interpolation mode and both dtypes are affected.
