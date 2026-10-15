# `tf.linalg.tridiagonal_solve` rejects NumPy inputs with an internal `AttributeError`

**Component:** TensorFlow 2.21.0 (CPU), `tf.linalg.tridiagonal_solve` in eager mode;
`tensorflow/python/ops/linalg/linalg_impl.py` (`_tridiagonal_solve_compact_format`,
`_tridiagonal_solve_matrix_format`, `_tridiagonal_solve_sequence_format`).
**Severity:** the op is **unusable with array-like inputs** - passing the documented `diagonals`/`rhs`
as NumPy arrays (or Python lists) raises `AttributeError: 'tuple' object has no attribute 'rank'`
instead of solving the system.  Every other TensorFlow linalg op accepts NumPy arrays, including the
sibling `tf.linalg.tridiagonal_matmul`, which parses the same `[3, M]` compact diagonals, and
`tridiagonal_solve` itself works with `tf.constant(...)` or under `tf.function`.
**Status:** present in 2.21.0 in eager mode.  The Python helper reads `diagonals.shape.rank` and
`rhs.shape.rank` directly, which only exists on `tf.TensorShape`; array-likes are never converted
first.  Graph mode (`tf.function`) converts the inputs at the op boundary and works.
**Found with:** the wave-374/375 Rondo TensorFlow linalg identity/consistency sweep (self-contained
oracles: `P @ L @ U == A`, `S @ S == A`, Moore-Penrose conditions, the three `diagonals_format`
variants must agree, `det`/`slogdet`/`logdet` consistency); certified reproducer `repro.py` in this
directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 21 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf
compact = np.array([[0.0, 1.0, 1.0], [3.0, 3.0, 3.0], [1.0, 1.0, 0.0]])   # [3, M] = (super, diag, sub)
rhs = np.array([[1.0], [0.0], [0.0]])

tf.linalg.tridiagonal_solve(compact, rhs)               # AttributeError: 'tuple' object has no
                                                        # attribute 'rank'
tf.linalg.tridiagonal_solve(compact, rhs, diagonals_format="matrix")     # same (with a dense matrix)
tf.linalg.tridiagonal_solve([du, d, dl], rhs, diagonals_format="sequence")
                                                        # AttributeError: ... 'is_compatible_with'

tf.linalg.tridiagonal_solve(tf.constant(compact), tf.constant(rhs))      # works
tf.linalg.tridiagonal_matmul(compact, rhs)                               # works (sibling op, numpy in)
```

The traceback localises the defect to TensorFlow's own Python code:

```
File ".../tensorflow/python/ops/linalg/linalg_impl.py", line 624, in _tridiagonal_solve_compact_format
    diags_rank, rhs_rank = diagonals.shape.rank, rhs.shape.rank
AttributeError: 'tuple' object has no attribute 'rank'
```

Complete input-type matrix at `diagonals_format="compact"` (the default):

| `diagonals` | `rhs` | result |
|---|---|---|
| NumPy | NumPy | `AttributeError: ... 'rank'` |
| NumPy | `tf.Tensor` | `AttributeError: ... 'rank'` |
| `tf.Tensor` | NumPy | `AttributeError: ... 'rank'` |
| Python list | NumPy | `AttributeError: ... 'rank'` |
| `tf.Tensor` | `tf.Tensor` | works, matches `numpy.linalg.solve` to 2.8e-17 |
| `tf.function(diags, rhs)` with tensors | | works |

The same holds for `diagonals_format="matrix"` (`AttributeError: ... 'rank'`) and
`"sequence"` (`AttributeError: 'tuple' object has no attribute 'is_compatible_with'`), so all three
documented formats are affected.

## Root cause

`tf.linalg.tridiagonal_solve` dispatches to pure-Python helpers that inspect the *static* shape of
their arguments:

```python
# tensorflow/python/ops/linalg/linalg_impl.py
def _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs, conjugate_rhs,
                                      partial_pivoting, perturb_singular, name):
  """Helper function used after the input has been cast to compact form."""
  diags_rank, rhs_rank = diagonals.shape.rank, rhs.shape.rank      # line 624
  ...
```

`tf.TensorShape.rank` exists; a NumPy `shape` tuple (or a Python list) has no `.rank`, and the
`is_compatible_with` call in the sequence path has the same problem.  TensorFlow's public op layer
converts array-likes via `ops.convert_to_tensor` before calling into helpers - here that conversion
happens only for `tf.function`/graph execution, not for eager calls, and the sibling
`tf.linalg.tridiagonal_matmul` does convert (it accepts the identical NumPy arguments).

## Suggested fix

Convert the inputs (or use a rank helper that accepts array-likes) before the shape inspection:

```python
  diagonals = ops.convert_to_tensor(diagonals)
  rhs = ops.convert_to_tensor(rhs)
  diags_rank, rhs_rank = diagonals.shape.rank, rhs.shape.rank
```

in `tf.linalg.tridiagonal_solve` (or at the top of each helper), and likewise replace
`diagonals.shape.is_compatible_with(...)` with the tensor-typed equivalent.  A regression test is the
compact-format call with NumPy `diagonals` and `rhs`, which must return the same result as the
`tf.constant` form.

## Impact

* NumPy is the default interchange format in TensorFlow code (`TensorLike` inputs are accepted by
  essentially every `tf.linalg`/`tf.math` op), so the natural call
  `tf.linalg.tridiagonal_solve(np_diags, np_rhs)` fails; users must remember to wrap both arguments
  in `tf.constant` (or run inside `tf.function`) - and the error they get is an internal
  `AttributeError` with a TensorFlow source line, not a validation message naming the argument.
* The failure is inconsistent within the module: `tridiagonal_matmul`, `solve`, `det`, `inv`,
  `matmul` and `diag_part` accept the same NumPy inputs (recorded in `evidence.txt`), and the
  sequence format fails with a different internal `AttributeError`, which makes the behaviour look
  accidental rather than documented.
* No silent wrong values: the op either works exactly (2.8e-17 against `numpy.linalg.solve`) or
  raises.

## Controls (all in `repro.py`, 21 checks, 0 failing on each of 2 replays)

* **Tensor control** - the identical data passed as `tf.constant` solves the system and matches
  `numpy.linalg.solve` to `2.8e-17`, for all three `diagonals_format` values.
* **Graph control** - `tf.function` around the same call returns the correct result, so the defect is
  the eager-mode input handling, not the kernel.
* **Sibling control** - `tf.linalg.tridiagonal_matmul(np_diags, np_rhs)` with the same compact
  diagonals returns `A @ b` to `2.2e-16`.
* **Module control** - `tf.linalg.solve`, `det`, `inv`, `matmul` and `diag_part` accept NumPy inputs.
* **Data control** - the failure follows the *container type*: the same array as a NumPy object raises
  while the same bytes as a `tf.constant` solve.
* **Failure-mode control** - the raised class is `AttributeError` with TensorFlow's own
  `linalg_impl.py` frame in the traceback, i.e. an internal code path, not the documented argument
  validation.

## Novelty

Searched the TensorFlow tracker (issues and PRs) on 2026-09-13 for
`tridiagonal_solve numpy`, `tridiagonal_solve AttributeError`, `_tridiagonal_solve_compact_format`,
`tridiagonal_solve rank` and `tridiagonal_solve array`: **zero** hits for each.  The campaign's own TF
linalg drafts are TF-020 (XLA `eigvalsh`), TF-024 (XLA `cholesky`), TF-025 (`inv`/`solve` det
overflow), TF-026 (`tf.linalg.qr`) and TF-027 (`tf.linalg.expm`) - all different ops; the wave-374/375
identity sweep that produced this witness found `lu`, `lu_solve`, `sqrtm` (except nilpotent matrices,
which have no square root), `pinv` (all four Moore-Penrose conditions), `eig`, `svd`, `matrix_rank`,
`triangular_solve` and the `det`/`slogdet`/`logdet` family consistent, and only this input-conversion
defect.

## Environment

```
tensorflow 2.21.0 (CPU), numpy 2.5.1, Python 3.12.13, WSL2 - see collect_env.txt and evidence.txt
```

## Anticipated objections and responses

* **"The docstring says `diagonals` is a `Tensor`."**  TensorFlow's universal convention (and the
  `TensorLike` annotation used throughout the API) is that array-likes are converted - the sibling
  `tridiagonal_matmul` in the same module documents the same `Tensor` type and accepts NumPy.  The
  error is an internal `AttributeError`, not a deliberate rejection.
* **"Wrap it in `tf.constant`."**  That is the workaround; the point is that the op silently diverges
  from every other linalg op and from its own sibling, and fails with a confusing internal error.
* **"It works in `tf.function`."**  Graph mode converts the inputs at the op boundary, which is
  exactly the conversion the eager path is missing; users calling the op directly from NumPy (the
  common eager style) hit the bug.
* **"This is only an error-message difference."**  No result is produced for a valid input: the
  computation is not performed at all, and the failure is an internal `AttributeError` in
  TensorFlow's Python code rather than a documented `ValueError`/`TypeError`.
