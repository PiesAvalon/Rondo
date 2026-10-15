# `ops.huber_loss` computes in float32 for float64 inputs

**Component:** MindSpore 2.9.0 (CPU), `mindspore.ops.huber_loss` (the `HuberLoss` kernel, all
reductions).
**Severity:** silent precision loss - with float64 inputs the returned tensor reports `Float64` but
its value is bit-for-bit the **float32** computation widened back for the default `reduction='mean'`
(and for `'sum'`):

```python
ops.huber_loss(x_float64, y_float64, delta=1.0) == ops.huber_loss(x_float32, y_float32, delta=1.0).astype(float64)
```

Against a float64 NumPy reference the error is `~1e-8` relative (`5.9e-09` at `mean`, `3.1e-08` at
`sum` for a 6x7 witness with values ~1e3; `~4e-8` for `reduction='none'`), while the same-family
`ops.smooth_l1_loss` on the same inputs is **float64-exact** (`0.0`).
**Status:** present in 2.9.0 on CPU, in PyNative and `ms.jit`; the bit-identity holds for
`delta` in {1, 2, 10} and both reductions, and `delta=0.1` is not bit-identical but still only
float32-accurate.
**Found with:** the wave-380/381 Rondo float32-dispatch oracle (the class behind MS-035/MS-038: a
float64 call bit-identical to the float32 call); certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 72 checks).

---

## Summary

| witness (6x7, values ~1e3) | `ops.huber_loss` float64 | NumPy float64 reference | relative error | bit-identical to the fp32 call |
|---|---|---|---|---|
| `delta=1.0, reduction='mean'` | `1093.86779785` | `1093.86780432` | `5.91e-09` | **True** |
| `delta=1.0, reduction='sum'` | `45942.44921875` | `45942.44778147` | `3.13e-08` | **True** |
| `reduction='none'` | elementwise | elementwise | `~4.2e-08` | False (still fp32-accurate) |
| `delta=0.1, reduction='mean'` | - | - | `1.84e-08` | True |
| control: `ops.smooth_l1_loss(reduction='mean')` | same inputs | - | **`0.0`** | False (float64) |

For every `(scale, delta in {1,2,10}, reduction in {mean,sum})` combination tested the float64 result
is bit-identical to the float32 call after widening, while `ms.jit` returns exactly the eager value.

## Root cause

The `HuberLoss` kernel computes the elementwise Huber loss and its reduction in float32 while the
output tensor keeps the input's `Float64` dtype; the float32 rounding is visible both as a
`~1e-8` relative error against an fp64 reference and as bit-identity with an explicit float32 call.
This is the same dispatch defect the campaign already filed for `ops.matrix_power` (MS-035) and
`ops.lu_solve` (MS-038), and the same class as the earlier MS mse/l1-loss records; `ops.smooth_l1_loss`
in the same module is float64-exact, which localises it to `HuberLoss`.

## Impact

* Any float64 Huber loss silently carries float32 rounding: the returned scalar/gradient looks
  correctly typed but is no more accurate than a float32 model, and for sums of many elements the
  absolute error grows with the batch (`~1e-5` absolute at a sum of `4.6e4`).
* The dtype metadata says `Float64`, so mixed-precision tooling cannot detect the loss.

## Suggested fix

Dispatch `HuberLoss` on the input dtype (or compute in float64 and cast on output) as
`ops.smooth_l1_loss` already does.  Regression test: `ops.huber_loss(x64, y64, delta=1.0)` must not be
bit-identical to the float32 call and must match the float64 reference to ~1e-15.

## Controls (all in `repro.py`, 72 checks, 0 failing on each of 2 replays)

* **Sibling control** - `ops.smooth_l1_loss(reduction='mean')` on the same inputs is float64-exact
  (`0.0` error), so the module does contain dtype-correct kernels.
* **Bit-exactness control** - the float64 `huber_loss` value equals the float32 call widened back for
  `delta` in {1, 2, 10} and both reductions.
* **Reduction control** - the defect covers `'mean'` and `'sum'`; `'none'` is not bit-identical but
  still only float32-accurate (`~4e-8`), i.e. the elementwise path is narrowed too.
* **Lane control** - `ms.jit` returns exactly the eager value.
* **Reference control** - every comparison is against a NumPy float64 implementation of the
  documented Huber formula, not another framework's output.

## Novelty

Searched the MindSpore tracker (GitHub mirror) on 2026-09-14 for `huber_loss`, `huber_loss float64`
and `huber_loss precision`: **zero** hits.  The campaign's wave-12 record covers
`mse_loss`/`l1_loss`/`smooth_l1_loss` (a different draft, and `smooth_l1_loss` is fp64-exact here) and
wave-232 covers the torch loss-offset controls; no record covers MindSpore's `huber_loss`.

## Environment

```
mindspore 2.9.0 (CPU, PyNative and GRAPH), numpy 1.26.4, Python 3.12.13, WSL2 - see collect_env.txt
```
