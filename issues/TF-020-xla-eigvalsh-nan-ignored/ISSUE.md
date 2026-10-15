# Under XLA, `tf.linalg.eigvalsh` ignores off-diagonal NaN/inf entries and returns the diagonal as the spectrum

> **Scope note (wave 357).**  The same "off-diagonal dropped, diagonal returned" behaviour also
> occurs for **large finite** inputs (`a >= 1e19` fp32 / `>= ~1e154` fp64) and that case is
> **already reported upstream as tensorflow#117807 (open)** - e.g. `[[1e20, 5e19], [5e19, 1e20]]`
> returns `[1e20, 1e20]` under XLA - with the related #118380 (open).  This draft therefore claims
> only the non-finite trigger, which neither issue covers; the finite checks in `repro.py` are kept
> as evidence that both triggers are the same root cause, and this report should be cross-referenced
> with #117807 (and may be a duplicate of it in a maintainer's view).

**Component:** TensorFlow 2.21.0, CPU: `tf.linalg.eigvalsh` inside `tf.function(..., jit_compile=True)`
(XLA), `float32` and `float64`.  Eager is the control lane and raises.
**Severity:** silent wrong result.  For a matrix whose self-adjoint decomposition is invalid (a NaN or
inf in the off-diagonal entries) XLA does not fail and does not propagate the NaN: it returns a
**finite, plausible spectrum equal to the matrix diagonal**, as if the off-diagonals were zero.  The
value passes `tf.debugging.check_numerics` and `tf.debugging.assert_all_finite` inside the same
compiled graph, so a downstream PCA / whitening / log-determinant computation has no way to notice.
**Status:** present in TensorFlow 2.21.0 on CPU (`XLA_CPU_JIT`).  Eager raises
`InvalidArgumentError: Self-adjoint eigen decomposition was not successful. The input might not be
valid.` for the NaN/inf tensors; valid inputs agree between the lanes at moderate magnitude.
**The trigger does not need a non-finite input, but that variant is already known:** for the finite
matrix `A = [[2a, a], [a, 2a]]` (exact eigenvalues `{a, 3a}`) the XLA lane is exact up to `a = 1e153`
and returns the **diagonal** `[2a, 2a]` from `a = 1e154` on - the same defect reached through an
overflow instead of a NaN.  That finite-input manifestation is **tensorflow#117807 (open)** (its
reproducer is the same family, fp32 from `1e19`) with the related #118380 (open), so it is recorded as
evidence here rather than claimed as new.
**Found with:** the wave-212 XLA linalg validity grid (Reference/eager vs Fresh/XLA over degenerate
matrices, the oracle that produced TF-018/TF-019 and PT-026); certified reproducer `repro.py` in this
directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 38 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf

M = tf.constant([[1., np.nan, 0.], [np.nan, 1., 0.], [0., 0., 1.]], tf.float64)

tf.linalg.eigvalsh(M)                                   # InvalidArgumentError
tf.function(lambda z: tf.linalg.eigvalsh(z), jit_compile=True)(M).numpy()
                                                        # [1. 1. 1.]   <- exactly the diagonal

# with a distinct diagonal the XLA result is that diagonal:
tf.function(lambda z: tf.linalg.eigvalsh(z), jit_compile=True)(
    tf.constant([[1., np.nan, 0.], [np.nan, 4., 0.], [0., 0., 9.]], tf.float64)).numpy()
                                                        # [1. 4. 9.]
```

Measured (float64; float32 behaves identically):

| input | eager | XLA (`jit_compile=True`) |
|---|---|---|
| `[[1, nan],[nan, 1]]` | `InvalidArgumentError` | `[1.0, 1.0]` |
| `[[1, nan, 0],[nan, 1, 0],[0,0,1]]` | `InvalidArgumentError` | `[1.0, 1.0, 1.0]` |
| 4x4 with one NaN off-diagonal pair | `InvalidArgumentError` | `[1.0, 1.0, 1.0, 1.0]` |
| NaN in a row and column (numbers distinct) | `InvalidArgumentError` | `[1.0, 2.0, 3.0]` |
| distinct diagonal with NaN off-diagonal | `InvalidArgumentError` | `[1.0, 4.0, 9.0]` |
| `inf` off-diagonal | `InvalidArgumentError` | `[1.0, 1.0, 1.0]` |
| `inf` on the diagonal | `[nan, nan, nan]` | `[1.0, 1.0, inf]` |
| all entries NaN | `InvalidArgumentError` | `[nan, nan, nan]` |
| symmetric, singular symmetric, non-symmetric, diagonal, `1e30` diagonal | values | same values (fp64 to 1e-9, fp32 to 1e-4) |

Two further observations from the reproducer:

* the all-NaN matrix returns all-NaN under XLA, i.e. a NaN **on the diagonal** survives while an
  off-diagonal NaN is dropped - the XLA path is not simply "NaN-propagating";
* `tf.debugging.check_numerics(eigvalsh(z))` and `tf.debugging.assert_all_finite(eigvalsh(z))` inside
  the compiled function both pass and return `[1.0, 1.0, 1.0]`, so the standard defensive checks in a
  compiled graph cannot detect the wrong result.

Finite-input trigger (`A = [[2a, a], [a, 2a]]`, exact eigenvalues `{a, 3a}`, all finite):

| `a` | exact | eager | XLA (`jit_compile=True`) |
|---|---|---|---|
| `1e100` | `[1e100, 3e100]` | exact | exact |
| `1e153` | `[1e153, 3e153]` | exact | exact |
| `1e154` | `[1e154, 3e154]` | exact | **`[2e154, 2e154]`** (the diagonal) |
| `1e155` | `[1e155, 3e155]` | exact | **`[2e155, 2e155]`** |
| `1e200` | `[1e200, 3e200]` | exact | **`[2e200, 2e200]`** |

`numpy.linalg.eigvalsh` returns `{a, 3a}` at every scale, `jnp.linalg.eigvalsh` (eager and jit) returns
`{a, 3a}` at `1e200` as well, and the same TensorFlow XLA lane is exact at `1e153` - so the answer is
reachable both inside TF and through the shared XLA kernels, and the switch at `a = 1e154` (where
`a**2` reaches the fp64 maximum) pins it to an overflowing intermediate in the TF->XLA lowering.

## Root cause

Eager `tf.linalg.eigvalsh` is backed by the `SelfAdjointEigV2` kernel, which runs a self-adjoint
Eigen decomposition and raises when it fails (the error text above).  The XLA lowering does not carry
that validity check: the compiled `Eigh`/`SelfAdjointEigV2` computation proceeds and, for these
inputs, produces the diagonal - it behaves as if the off-diagonal entries were zero (compare the
control in the reproducer where replacing the NaN by `0` makes both lanes return the diagonal, which
is then the correct spectrum).  Because the returned vector contains no NaN, the failure is not
visible through `check_numerics`/`assert_all_finite` either.

## Impact

* A covariance, correlation or Hessian matrix that contains a single corrupt (NaN/inf) entry silently
  yields a plausible eigenvalue vector instead of an error.  PCA, whitening, condition-number checks,
  positive-definiteness tests and log-determinant terms built on those eigenvalues proceed with a
  spectrum that is not the spectrum of the input.
* The eager and compiled lanes of the same program disagree qualitatively: eager stops the program,
  XLA continues.  A model trained eagerly and the same model compiled therefore behave differently
  exactly when its inputs are already corrupted.
* The value cannot be screened for: it is finite and its largest/smallest entries are the diagonal
  entries, so range checks and finiteness checks pass.

## Suggested fix

Validate the input in the XLA lowering the way the eager kernel does (for example by checking
finiteness of the operand and raising/returning a status through `XlaRuntimeError`, or by lowering
`SelfAdjointEigV2` with a `chlo`/`mhlo` check that fails instead of computing on non-finite entries).
A regression test needs no reference library:

```python
m = tf.constant([[1., np.nan], [np.nan, 1.]], tf.float64)
with self.assertRaises(tf.errors.InvalidArgumentError):
    tf.function(tf.linalg.eigvalsh, jit_compile=True)(m)
```

## Controls (all in `repro.py`, 69 checks, 0 failing on each of 2 replays)

* Five **valid** input families (symmetric, singular symmetric, non-symmetric, diagonal-only and
  `1e30`-diagonal matrices) at both widths: the lanes agree (fp64 to `1e-9`, fp32 to `1e-4`), so this
  is the dropped validity check and not an accuracy difference.
* The eager lane is asserted to raise `InvalidArgumentError` for every NaN/inf case, in both widths.
* The XLA outputs are asserted to equal the matrix diagonal exactly (values chosen distinct, e.g.
  `[1, 4, 9]`), which identifies what the compiled computation returns.
* The all-NaN control shows a diagonal NaN survives, so the behaviour is not generic NaN propagation.
* The control where the off-diagonal NaN is replaced by `0` makes both lanes agree on the diagonal,
  confirming that XLA behaves as if the off-diagonals were zero.
* `check_numerics` and `assert_all_finite` are exercised inside the compiled function and asserted to
  pass; `float32` and `float64` are both covered.

## Novelty

The NaN/inf trigger has no tracker hit (the searches in the wave-213 novelty check are unchanged).
The **finite** large-magnitude trigger that the wave-357 finite-extreme-value sweep added is a
different story and is *not* novel: **tensorflow#117807 (open) "`tf.linalg.eigvalsh` returns wrong
eigenvalues under XLA for large-magnitude matrices"** reports exactly this - XLA returns the diagonal
from `v ~ 1e19` (fp32), with `[[1e20, 5e19],[5e19, 1e20]]` -> `[1e20, 1e20]` - and **#118380 (open)**
is the same report for a float32 matrix; #117864 (open) is the complex-input case and #48221/#47661
(closed) are eager failures.  The earlier wave-213 note described #117807/#118380 as "accuracy on
large-magnitude valid matrices", which understated the overlap: they are the same "off-diagonal
ignored" observable, so the finite variant is recorded as known (see the ledger row
`wave357-eigvalsh-finite-known`) and this draft is deliberately limited to the non-finite trigger, to
be filed with a cross-reference to #117807.

TensorFlow tracker search for `eigvalsh nan`, `eigvalsh xla`, `eigh nan`, `eigvalsh wrong` and
`self-adjoint eigen decomposition invalid` returns:

* [#117807](https://github.com/tensorflow/tensorflow/issues/117807) (open) - `eigvalsh` returns wrong
  eigenvalues under XLA for **large-magnitude** matrices, and
  [#118380](https://github.com/tensorflow/tensorflow/issues/118380) (open) - large symmetric matrix,
  both about numerical accuracy on **valid** input;
* [#117864](https://github.com/tensorflow/tensorflow/issues/117864) (open) - complex input under XLA;
* [#48221](https://github.com/tensorflow/tensorflow/issues/48221) (closed, 2021) - eager `eigh`/
  `eigvalsh` failing inside a loss function on ordinary data (no XLA, no NaN);
* [#47661](https://github.com/tensorflow/tensorflow/issues/47661) (closed) - eager `eigh` of a 26x26
  zero matrix on GPU;
* [#118378](https://github.com/tensorflow/tensorflow/issues/118378) - the campaign's own earlier
  finding that XLA `tf.linalg.inv` returns finite garbage for singular input, which is a different op
  and a different observable (the campaign recorded it as known in wave 194).

No item describes a non-finite **off-diagonal** entry being ignored and the diagonal being returned as
the spectrum, nor the fact that `check_numerics` cannot detect it.  The campaign's other XLA-validity
drafts are TF-018 (`tf.gather`/`embedding_lookup` index clamping) and TF-019
(`tf.image.non_max_suppression` threshold range), other ops and other checks.

## Environment

TensorFlow 2.21.0, CPU (`XLA_CPU_JIT`), Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"The input is invalid, so the result is undefined - garbage in, garbage out."**  Eager refuses it
  and XLA returns a *plausible finite answer that is not flagged anywhere*, including by TF's own
  `check_numerics`/`assert_all_finite`.  TensorFlow's XLA backend is expected to preserve NaN
  semantics or fail; silently dropping an off-diagonal NaN and returning the diagonal is neither.
  TF-018 and TF-019 are the same class for index clamping and for a threshold range, and both were
  filed against XLA's missing checks.
* **"XLA cannot raise."**  It can surface a failing status (`XlaRuntimeError`) and other XLA kernels do
  validate; alternatively the lowering can propagate the NaN like every IEEE operation would.
* **"Only a corrupt matrix triggers it."**  That is exactly when a program must not silently continue:
  the failure mode is a plausible spectrum, and the corpus of corrupt covariance matrices is a real
  training/scoring scenario.
* **"It is the same as your TF inv finding (#118378)."**  That one is about `tf.linalg.inv` on
  singular input; this is `eigvalsh` on non-finite input, a different op, kernel and observable (and a
  different lane divergence: here eager raises for *all* tested non-finite inputs while `inv` returned
  finite values).
