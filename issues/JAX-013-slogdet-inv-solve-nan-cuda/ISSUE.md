# On the CUDA backend, `jnp.linalg.inv`/`solve` (and `det`/`slogdet` for `n >= 3`) silently replace NaN entries with `0.0`

**Component:** JAX 0.10.2 / jaxlib 0.10.2, **CUDA backend** (XLA:GPU; RTX 4090, CUDA 13.0).  The
reference lanes are the **CPU backend** of the same build (run in its own process) and NumPy.
`jnp.linalg.det`, `jnp.linalg.slogdet`, `jnp.linalg.inv`, `jnp.linalg.solve`, `float32` and
`float64`.
**Severity:** silent wrong result.  A matrix that contains NaN produces **ordinary finite numbers**
from the LU-based linalg functions instead of NaN.  `jnp.linalg.inv(A)` is *bit-for-bit* the inverse
of the matrix obtained by replacing the NaN with `0.0`; the determinant, `slogdet` and `solve`
likewise.  A NaN produced anywhere upstream (a collapsed loss, a corrupt input, an uninitialised
buffer) becomes a plausible finite determinant/inverse/solution, and the divergence is CUDA-only, so
a CPU-only test suite stays green.
**Status:** present in JAX 0.10.2 / jaxlib 0.10.2 on the CUDA backend.  The same build's CPU backend
propagates the NaN.  The observable is size-dependent (see the table), which shows it is a value-path
defect and not a documented "CUDA ignores NaN" convention.
**Found with:** the wave-260 JAX device differential (XLA:CPU vs XLA:GPU value comparison over a
linalg/edge grid) and the wave-261 characterisation; certified reproducer `repro.py` in this
directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 108 checks, 2 process-isolated replays).

---

## Summary

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

A  = jnp.array([[1.0, jnp.nan], [2.0, 3.0]])   # GPU backend
A0 = jnp.array([[1.0, 0.0],       [2.0, 3.0]]) # the same matrix with NaN -> 0.0

jnp.linalg.solve(A,  jnp.ones(2))   # CUDA: [1.0, -0.3333333333333333]   CPU: [nan, nan]
jnp.linalg.solve(A0, jnp.ones(2))   #        [1.0, -0.3333333333333333]   (identical)
jnp.linalg.inv(A)                   # CUDA: [[1.0, 2.7755575615628914e-17], [-0.6666666666666666, 0.3333333333333333]]
jnp.linalg.inv(A0)                  #        (identical, bit for bit)
```

| shape | matrix (NaN position) | `inv` / `solve` | `det` | `slogdet` |
|---|---|---|---|---|
| **2x2** | `[[1, nan], [2, 3]]` | **finite (wrong)** = value of `[[1,0],[2,3]]` | nan (correct) | nan (correct) |
| **3x3** | `[[1,2,3], [4,nan,6], [7,8,10]]` | **finite (wrong)** | nan (correct) | **finite (wrong)** = `log(52)` |
| **4x4** | `[[4,1,2,0.5], [1,3,nan,1], [2,0.5,5,1], [0.5,1,1,3]]` | **finite (wrong)** | **finite (wrong)** = `109.5` | **finite (wrong)** = `4.695924549256556` |
| 2x2 | `[[nan, 0], [0, 1]]` | `[inf, nan]` (wrong; the substituted matrix is singular) | nan | nan |

The substituted results are not merely close to the `NaN -> 0` values, they are **bit-identical** to
them (transcript in `evidence.txt`), which identifies the mechanism exactly.

## Root cause

`det`/`slogdet`/`inv`/`solve` lower to an LU factorization; on the CUDA backend the factorization is
performed by cuSOLVER (`getrf`), and an entry that is NaN is used as `0.0` inside it - the same
behaviour PyTorch reports for `torch.linalg.slogdet` on CUDA in
[pytorch#173638](https://github.com/pytorch/pytorch/issues/173638) ("computes it as if there was a 0
in the NaN's place").  Because the NaN never reaches the returned values, every downstream quantity
is a plausible finite number:

* `inv`/`solve` divide by the substituted pivots and return the solution of the zeroed system;
* `slogdet` returns `log|det|` of the zeroed matrix (3x3: `3.951243718581427 = log(52)`;
  4x4: `4.695924549256556 = log(109.5)`);
* `det` returns the determinant of the zeroed matrix (`109.5` for the 4x4 witness);
* when the substitution makes the matrix singular (`[[nan,0],[0,1]] -> [[0,0],[0,1]]`) the result
  turns into `inf` (and a `0/0` NaN) rather than staying NaN.

The defect is **not** a blanket "CUDA drops NaN": `jnp.linalg.svd`, `pinv` and `eigvalsh` propagate
NaN on CUDA exactly as they do on CPU (asserted in the reproducer), and for 2x2 inputs `det` and
`slogdet` still propagate NaN - those paths do not go through the same LU value path.  The
size-dependence is itself part of the report: the same call is NaN-correct on one shape and silently
substituted on the next.

## Oracles (Rondo model)

* **Reference** - the same program on the **CPU backend in its own process**
  (`JAX_PLATFORM_NAME=cpu`), plus NumPy: every witness asserts an all-NaN/finite reference in both.
* **Fresh** - a freshly jitted CUDA callable (`jax.jit(op)(A)`) is compared with the eager CUDA
  result for every witness/op: same substituted value.
* **Persistent** - one `jax.jit(jnp.linalg.inv)` callable is reused for 200 clean `3x3` calls and
  then handed the NaN witness: it still substitutes `0.0`, so this is not a cold-compile artifact.
* **Process isolation** - `repro.py --replays 2` runs the whole suite in two child processes, each
  with its own CUDA context and its own freshly compiled programs; 2/2 replays pass with 0 failing
  checks.

## Controls (all asserted in `repro.py`)

* NumPy and the CPU backend propagate NaN for every witness and every op (the reference lanes).
* `svd`, `pinv`, `eigvalsh` propagate NaN on CUDA for every witness (LU-path controls).
* 2x2 `det`/`slogdet` propagate NaN on CUDA (so the substitution is size/path dependent, not a
  convention).
* The substituted value is asserted to equal the `NaN -> 0` matrix's value within 1e-9 relative and
  to contain no NaN.
* The `float32` lane repeats the `inv`/`solve` substitution.
* The minimal witness is asserted directly (`solve([[1,nan],[2,3]], 1)` is finite and equals
  `solve([[1,0],[2,3]], 1)`).

## Related observation (not part of this report's claim)

`jnp.linalg.matrix_rank` returns **0** for a matrix containing NaN on *both* backends, because
`NaN > tol` is `False` and the rank is a sum of booleans; NumPy raises `LinAlgError`.  That is a
different root cause (a comparison, not the LU substitution) and is recorded here only so it is not
mistaken for part of the CUDA issue.

## Impact

* **Silent**: the outputs are finite and have the right shape/dtype; only the values are those of a
  different matrix.  A finiteness or `torch.isfinite`-style guard does not fire, and a residual check
  against the original matrix can look acceptable for mildly conditioned systems.
* **CUDA-only**: the CPU backend is correct, so the defect appears only in GPU runs and only in the
  LU-based functions.  CPU/GPU comparison tests are exactly what is needed to see it.
* **NaN is the normal end state of a diverged computation**, so the substitution turns the clearest
  possible corruption signal into a plausible number.

## Suggested fix

Make the LU-based lowering NaN-safe on the GPU path: propagate a non-finite input through
`getrf`/the custom call (e.g. detect a non-finite input and return NaN results, or make the kernel
emit NaN instead of a zero pivot for a NaN entry).  A regression test in JAX's linalg suite:

```python
A = jnp.array([[1.0, jnp.nan], [2.0, 3.0]])
assert jnp.isnan(jnp.linalg.solve(A, jnp.ones(2))).all()   # currently finite on CUDA
assert jnp.isnan(jnp.linalg.inv(A)).all()                  # currently finite on CUDA
```

## Novelty

JAX tracker searches for `linalg inv nan`, `det nan`, `slogdet`, `solve nan gpu`, `linalg nan
propagation` and XLA searches for `getrf nan`, `LU nan decomposition`, `triangular_solve nan` return
no report of this observable.  The nearest items are:

* [pytorch#173638](https://github.com/pytorch/pytorch/issues/173638) - `torch.linalg.slogdet` does
  not propagate NaN on CUDA; the same cuSOLVER mechanism, but a different project and API, disclosed
  here rather than claimed as this report's own.  It is the reason this draft is filed as
  JAX-side/GPU-path rather than as a new library discovery.
* [jax#21122](https://github.com/jax-ml/jax/issues/21122) - `linalg.solve` produces NaNs on GPU but
  not CPU: the **opposite** direction (GPU invents NaNs, CPU does not).
* [jax#33707](https://github.com/jax-ml/jax/issues/33707) - SVD does not propagate NaNs for batch
  sizes > 2: a different operator (`svd`), and this campaign's reproducer asserts `svd` is one of the
  *correct* CUDA lanes.
* This campaign's TF-020 (XLA `eigvalsh` ignoring off-diagonal NaN) is the same "XLA GPU linalg
  ignores non-finite input" class in a different frontend and operator; the JAX `eigvalsh` lane is
  asserted correct here.

## Environment

JAX 0.10.2 / jaxlib 0.10.2, CUDA backend (`NVIDIA GeForce RTX 4090`, CUDA 13.0, driver 591.86) and
CPU backend for the reference lane, Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64,
`jax_enable_x64=True`.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"NaN handling in LAPACK/cuSOLVER is undefined, so any result is acceptable."**  Undefined
  ordering does not explain `inv(A) == inv(A0)` bit for bit: the NaN is *replaced by zero*, which is
  a specific value substitution.  The same build's CPU backend and NumPy return NaN, and the JAX
  documentation for these functions states no NaN caveat.
* **"Then it is a cuSOLVER/lib behaviour, not a JAX bug."**  The JAX report is about the JAX public
  API's observable, which is CUDA-only and inconsistent across shapes of the same call.  It can be
  fixed in JAX's lowering (guard the custom call) exactly as the CPU lane already behaves; PyTorch's
  equivalent report (#173638) was accepted as a framework issue for the same reason.
* **"Does the CPU backend have the same problem?"**  No: every witness asserts all-NaN `inv`/`solve`
  and NaN `det`/`slogdet` on the CPU backend, in its own process.  For rank-deficient matrices both
  NumPy and JAX can return `0.0` (the determinant is genuinely zero there), which is why every
  witness here is well conditioned.
* **"This requires a GPU to reproduce."**  True; `repro.py` prints the CPU reference lane and skips
  the CUDA assertions with a note when no GPU is present, so the package is still runnable on a
  CPU-only machine.
* **"Does the GPU lane produce NaN sometimes anyway?"**  Yes, for the smallest shapes - and that is
  asserted as a control, so the report does not overstate the scope.
