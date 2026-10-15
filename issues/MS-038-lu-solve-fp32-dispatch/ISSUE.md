# `ops.lu_solve` computes in float32 for float64 inputs

**Component:** MindSpore 2.9.0 (CPU, PyNative and GRAPH), `mindspore.ops.lu_solve`
(`LuSolve` kernel); the companion `ops.lu_unpack` is exact, which isolates the defect to the solve.
**Severity:** silent loss of float64 accuracy - the returned tensor reports `Float64` but its values
are bit-for-bit the **float32** computation widened back:

```python
ops.lu_solve(b_float64, LU_float64, pivots) == ops.lu_solve(b_float32, LU_float32, pivots).astype(float64)
```

On `A = [[1, 1], [1, 1+1e-6]]`, `b = A @ [1, 1]` the exact solution `[1, 1]` comes back as
`[1.04632568, 0.95367432]` (absolute error `4.6e-2`) although the packed factorisation is a valid
float64 LU (`ops.lu_unpack` reproduces it exactly) and `ops.matrix_solve` solves the same system to
`2.2e-10`.
**Status:** present in 2.9.0 in PyNative and `ms.jit` on CPU.  Every witness tested - well-conditioned,
integer, Hilbert and near-singular - shows the bit-identity with the float32 call.
**Found with:** the wave-378 Rondo MindSpore linalg-selector/identity sweep (the `lu_unpack`/`lu_solve`
pair checked cross-framework against `torch.linalg.lu_factor`/`torch.lu_unpack` and against
`ops.matrix_solve`); certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 30 checks).

---

## Summary

| witness | `ops.lu_solve` (float64 in) | exact solution | error | `ops.matrix_solve` error |
|---|---|---|---|---|
| `[[4,1,.5],[2,5,1],[.5,3,6]]` | `[1.0, 1.0, 0.99999988]` | `[1, 1, 1]` | `1.19e-07` | `0.0` |
| `[[2,1,1],[4,5,3],[6,10,12]]` | `[0.99999952, 1.0, 1.00000024]` | `[1, 1, 1]` | `4.77e-07` | `4.44e-16` |
| Hilbert 3x3 | `[0.99999857, 1.00000834, 0.99999189]` | `[1, 1, 1]` | `8.35e-06` | `1.03e-14` |
| `[[1,1],[1,1+1e-6]]` | `[1.04632568, 0.95367432]` | `[1, 1]` | **`4.63e-02`** | `2.22e-10` |
| `[[1,2,3],[4,5,6],[7,8,10+1e-6]]` | `[1.00000191, 0.99999571, 1.00000215]` | `[1, 1, 1]` | `4.29e-06` | `3.55e-15` |

For every row, `ops.lu_solve(float64)` is **bit-identical** to `ops.lu_solve(float32).astype(float64)`
(`np.array_equal` is `True`) while the dtype of the float64 call reads `Float64`; the float32 call
reads `Float32` and returns the same numbers.  `ms.jit` returns the same values as eager.

## Root cause

`ops.lu_solve` is documented (and named) as the float64-capable companion of `ops.lu_unpack`, and the
input `LU_data`/`b` float64 tensors are preserved in the output's *metadata* - but the kernel performs
the triangular substitutions in float32 and writes them back into a Float64 tensor.  The observable is
the same dispatch defect as the campaign's MS-035 (`ops.matrix_power` computing in float32 for
float64), with a bit-exact proof: widening the float32 result reproduces the float64 result exactly.
The packed factorisation itself is fine - `ops.lu_unpack` returns P/L/U identical to
`torch.lu_unpack`'s for the same data and `P @ L @ U == A` to 1e-16 - and `ops.matrix_solve`,
`ops.pinv`, `ops.lstsq` and `ops.matrix_solve(adjoint=True)` are float64-exact on the same systems.

## Suggested fix

Dispatch `LuSolve` on the input dtype (factor the float32 and float64 kernels, as the other linalg
ops do) instead of computing in float32 and casting to the requested dtype.  A regression test is any
of the witnesses above: `ops.lu_solve(b, LU, piv)` must not be bit-identical to the float32 call, and
the near-singular 2x2 must return `[1, 1]` to ~1e-10.

## Impact

* Any float64 LU solve silently degrades to float32: for a 2x2 system with `cond ~ 4e6` the answer is
  wrong in the second decimal, and for larger/more ill-conditioned systems the loss grows with the
  conditioning.  The dtype metadata says `Float64`, so downstream code (and mixed-precision checks)
  cannot see it.
* `lu_solve` is the standard way to reuse a factorisation across many right-hand sides (e.g. in
  implicit time stepping or Newton iterations); the factorisation is computed in float64 and then
  discarded at the solve step.
* `ops.matrix_solve` is float64-exact on the same systems, so the workaround is to avoid `lu_solve`
  entirely - which defeats the purpose of keeping a factorisation.

## Controls (all in `repro.py`, 30 checks, 0 failing on each of 2 replays)

* **Input control** - `ops.lu_unpack(LU, piv)` returns P/L/U with `P @ L @ U == A` to 1e-12 for every
  witness (and identical to `torch.lu_unpack`'s output for the same packed data), so the packed input
  is a genuine float64 factorisation.
* **Bit-exactness control** - the float64 and float32 calls are bit-identical for every witness, i.e.
  the defect is a dtype dispatch and not rounding inside an otherwise float64 algorithm.
* **Metadata control** - the float64 call reports `Float64` and the float32 call `Float32`, so the
  wrong precision is invisible in the returned tensor's dtype.
* **Sibling control** - `ops.matrix_solve`, `ops.matrix_solve(adjoint=True)`, `ops.pinv` and
  `ops.lstsq` are float64-exact on the same systems (≤ 1e-14, except the near-singular case at
  `2.2e-10`, which is fp64-consistent for `cond ~ 4e6`).
* **Lane control** - `ms.jit` returns exactly the eager (float32-quality) values.

## Novelty

Searched the MindSpore tracker (GitHub mirror `mindspore-ai/mindspore`, issues and PRs) on
2026-09-14 for `lu_solve`, `lu_solve float32`, `lu_solve precision` and `LU Solve float64`: **zero**
hits for each.  The campaign's nearest records are MS-035 (`ops.matrix_power` float32 dispatch - a
different op, same class) and the wave-374/375 TF `lu_solve` identity checks (a different framework);
no ledger record covers MindSpore's `lu_solve` or `lu_unpack`.

## Environment

```
mindspore 2.9.0 (CPU, PyNative and GRAPH), numpy 1.26.4, Python 3.12.13, WSL2
the packed LU factors were produced once with torch 2.12.1+cu130 `torch.linalg.lu_factor` (recorded in
evidence.txt) and are re-validated inside the reproducer with `ops.lu_unpack` - see collect_env.txt
```

## Anticipated objections and responses

* **"float32 is the default dtype."**  The inputs are float64 tensors (LU, pivots and `b` are all
  built as float64) and the output dtype reports `Float64`; `ops.matrix_solve` on the same matrices is
  float64-exact, so the dtype is not being ignored by design anywhere else in the module.
* **"The near-singular witness is ill-conditioned."**  That is what makes the float32 loss visible;
  the well-conditioned and integer witnesses still show `1e-7` errors (float32 rounding) where an
  fp64 solve gives `0` / `4e-16`, and the bit-identity with the float32 call is what proves the
  mechanism rather than conditioning.
* **"Use `matrix_solve`."**  That is the workaround, and it recomputes the factorisation - the point
  of `lu_solve` is to reuse one.
* **"The values are close enough."**  `4.6e-2` absolute on a system that solves exactly, together
  with a `Float64` label, is a silent precision defect of the same kind the campaign already filed for
  `matrix_power` (MS-035).
