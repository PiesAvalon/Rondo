# `jax.scipy.linalg.eigh` ignores its documented `lower` argument

**Component:** JAX 0.10.2 (CPU), `jax.scipy.linalg.eigh` (SciPy-compatibility wrapper),
`jax._src.scipy.linalg._eigh` -> `jax.lax.linalg.eigh`.
**Severity:** silently wrong eigenvalues for any input whose two triangles differ.  The wrapper
documents *"lower: if True (default) access only the lower portion of the input matrix. Otherwise
access only the upper portion"*, but it forwards to `lax.linalg.eigh(a, lower=lower)` while leaving
the lax-level `symmetrize_input=True` default in place, so the matrix is replaced by
`(a + a^H)/2` and the flag has no effect (unlike `jnp.linalg.eigh`, the wrapper exposes no
`symmetrize_input` argument, so the documented behaviour is unreachable):

```python
A = [[2., 1.], [0., 3.]]
jax.scipy.linalg.eigh(A, lower=True )[0]   # [1.792893, 3.207107]
jax.scipy.linalg.eigh(A, lower=False)[0]   # [1.792893, 3.207107]   identical
scipy.linalg.eigh(A, lower=True )[0]       # [2.0, 3.0]
scipy.linalg.eigh(A, lower=False)[0]       # [1.381966, 3.618034]
```

`[1.792893, 3.207107]` is the spectrum of `(A + A^T)/2` - *neither* SciPy convention.  The correct
values are reachable one level down (`jax.lax.linalg.eigh(A, lower=..., symmetrize_input=False)`
matches SciPy exactly for both settings), and the sibling wrappers
`jax.scipy.linalg.cholesky(lower=...)` and `jax.scipy.linalg.solve_triangular(..., lower=...)` honour
their flags, so the defect is a missing `symmetrize_input=False` in one wrapper.
**Status:** present in 0.10.2 and at master (`jax/_src/scipy/linalg.py` line 407:
`v, w = lax_linalg.eigh(a, lower=lower)`; `jax/_src/lax/linalg.py::eigh` still defaults
`symmetrize_input=True`).  Present in eager and under `jax.jit`.
**Found with:** the wave-373 Rondo "documented selector argument" sweep (the TF-028 / MS-036
`periodic`/`dtype` method applied to JAX's triangular-solve family); certified reproducer
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 46 checks).

---

## Summary

| witness | `jax.scipy.linalg.eigh(lower=True)` | `lower=False` | `scipy` `lower=True` | `scipy` `lower=False` |
|---|---|---|---|---|
| `[[1,0,0],[2,1,0],[3,2,1]]` | `[-0.5, 0.149219, 3.350781]` | same | `[-2.0, -0.701562, 5.701562]` | `[1.0, 1.0, 1.0]` |
| `[[2,1],[0,3]]` | `[1.792893, 3.207107]` | same | `[2.0, 3.0]` | `[1.381966, 3.618034]` |
| `[[2, 1+1j],[0, 3]]` (complex) | `[1.633975, 3.366025]` | same to 1e-12 | `[2.0, 3.0]` | `[1.0, 4.0]` |

The first row is the witness of the closed JAX issue **#9473** (`jnp.linalg.eigh` accessing the full
matrix); the JAX answer is the eigenvalue set of `(a + a^H)/2`, which is what
`symmetrize_input=True` is documented to do *at the `jnp`/`lax` level* - but the SciPy wrapper claims
the `lower` semantics and gives no way to turn the symmetrisation off.

Eigenpairs, not just eigenvalues: for `A = [[2,1],[0,3]]` and `lower=True`, the first returned
eigenpair has residual `4.6e-01` against the requested lower-triangle problem
(`A_L = tril(A) + tril(A)^T - diag`, `A_L v = lambda v`), `2.2e-16` against the fully symmetrised one,
and SciPy's `lower=True` pair resolves the lower-triangle residual to `1e-9`.

## Root cause

```python
# jax/_src/scipy/linalg.py  (master line 407)
@jit(static_argnames=('lower', 'eigvals_only', 'eigvals', 'type'))
def _eigh(a, b, lower, eigvals_only, eigvals, type):
  ...
  a, = promote_dtypes_inexact(jnp.asarray(a))
  v, w = lax_linalg.eigh(a, lower=lower)          # <- symmetrize_input defaults to True
  ...

# jax/_src/lax/linalg.py
def eigh(x, *, lower: bool = True, symmetrize_input: bool = True, ...):
  """...
     lower: If ``symmetrize_input`` is ``False``, describes which triangle of the input matrix to
       use. If ``symmetrize_input`` is ``False``, only the triangle given by ``lower`` is accessed;
       the other triangle is ignored and not accessed.
  """
  if symmetrize_input:
    x = symmetrize(x)                              # (x + x^H) / 2
  v, w = eigh_p.bind(x, lower=lower, ...)
```

`jnp.linalg.eigh(a, UPLO=..., symmetrize_input=False)` reaches the documented triangle behaviour (and
its docstring states that with the default `symmetrize_input=True` *both* triangles are used - so it
is not affected).  `jax.scipy.linalg.eigh` has no `symmetrize_input` parameter and its docstring
documents `lower` without any caveat, so the flag is dead code for every input.

## Suggested fix

Pass the SciPy semantics through:

```python
  v, w = lax_linalg.eigh(a, lower=lower, symmetrize_input=False)
```

(optionally exposing `symmetrize_input` as an extra keyword for users who want the current
behaviour).  `jax.lax.linalg.eigh(..., symmetrize_input=False)` already matches
`scipy.linalg.eigh(a, lower=...)` bit-for-bit in the tested cases, so the change is contained.

## Impact

* Porting SciPy code that relies on the documented triangle convention silently produces different
  eigenvalues and eigenvectors, with no error or warning.
* The convention exists precisely for inputs whose unused triangle is irrelevant or garbage (e.g. a
  triangular factor, a view into a larger buffer, or a matrix built with only one triangle filled in).
  LAPACK's `syevd`/`heevd` and SciPy read only the requested triangle; JAX reads both.
* `jax.scipy.linalg.eigh` is the documented SciPy-compatibility entry point, so a user cannot tell
  from the API that the parameter is inert; the workaround (`lax.linalg.eigh(..., symmetrize_input=False)`)
  requires knowing the internal flag.

## Controls (all in `repro.py`, 46 checks, 0 failing on each of 2 replays)

* **Flag control** - `lower=True` and `lower=False` return the same spectrum (bit-identical for real
  input, to 1e-12 for complex) for three witnesses, while `scipy.linalg.eigh` differs between the two
  settings by 1e-3 .. 20.
* **Mechanism control** - the returned spectrum equals `eigvalsh((a + a^H)/2)` to 1e-9 and differs
  from both triangle-selected spectra to > 1e-3.
* **Fix-reachability control** - `jax.lax.linalg.eigh(a, lower=..., symmetrize_input=False)` matches
  `scipy.linalg.eigh(a, lower=...)` to 1e-9 for both flags on all three witnesses.
* **Module control** - `jax.scipy.linalg.cholesky(lower=...)` and
  `jax.scipy.linalg.solve_triangular(..., lower=...)` match SciPy for both flags, so the missing
  argument is specific to `eigh`.
* **API control** - `jnp.linalg.eigh(A, UPLO='L'/'U', symmetrize_input=False)` matches NumPy, and with
  the default `symmetrize_input=True` it returns the fully symmetrised spectrum, which its docstring
  states explicitly (this API is *not* part of the report).
* **Symmetry control** - with a genuinely symmetric input every lane (JAX `lower=True/False`, SciPy
  `lower=True/False`) agrees to 1e-9, so the defect only shows when the two triangles differ.
* **Lane control** - `jax.jit(lambda a: jax.scipy.linalg.eigh(a, lower=True)[0])` equals the eager
  result and `lower=False` under `jit` equals `lower=True` under `jit`.

## Novelty

Searched the JAX tracker on 2026-09-13 (`scipy.linalg.eigh lower`, `eigh lower ignored`,
`eigh symmetrize_input`, `scipy.linalg.eigh wrong`): no report mentions
`jax.scipy.linalg.eigh`.  The related items are all about the **NumPy** API:
**#9473** (closed, 2022) "`jnp.linalg.eigh` accesses full matrix rather than lower part",
**#12687** (closed) "`jax.numpy.linalg.eigh` produces different results for full vs triangular
matrices", and **#12696** (open, docs/validation PR) "`jnp.linalg.eigh`: document `symmetrize_input` &
error when using with `UPLO`" - those were resolved by documenting `symmetrize_input` (and the
`UPLO`-with-`symmetrize_input=False` path works).  The SciPy wrapper was left out of that fix: its
docstring still promises triangle selection, its signature has no `symmetrize_input`, and the
`symmetrize_input=True` default is still hard-wired at master.  The campaign's earlier JAX linalg
records are JAX-018 (`slogdet` 2x2) and the finite-extreme sweeps; none touch `eigh`'s triangle
selection.

## Environment

```
jax 0.10.2 (CPU, x64 enabled), numpy 2.5.1, scipy 1.18.0, Python 3.12.13, WSL2
(the SciPy reference is imported in the same interpreter - see collect_env.txt and evidence.txt)
```

## Anticipated objections and responses

* **"`eigh` inputs must be Hermitian, so the triangles agree."**  The parameter exists exactly to
  support inputs where the unused triangle is not meaningful, which is standard LAPACK/SciPy usage
  (and the docstring promises it).  For strictly Hermitian input there is no difference - which is why
  this is a silent compatibility defect rather than a numerical one.
* **"This is the known `jnp.linalg.eigh` issue #9473."**  That issue (and #12687/#12696) is about the
  NumPy-compatibility API, which now documents `symmetrize_input` and honours `UPLO` when it is
  `False`.  This report is the SciPy-compatibility wrapper, which still documents triangle selection
  with no caveat and offers no way to disable symmetrisation; the fix is a different one-line change.
* **"`lower` is just unused metadata."**  It changes the mathematical problem that is solved: the
  returned eigenpairs do not satisfy the requested `A_L v = lambda v` (residual `0.46` in the 2x2
  witness) and differ from `scipy.linalg.eigh` by up to 20% of the spectrum.
* **"Use `lax.linalg.eigh`."**  That is the workaround, and it requires knowing an internal keyword
  (`symmetrize_input=False`) that the SciPy wrapper hides.
