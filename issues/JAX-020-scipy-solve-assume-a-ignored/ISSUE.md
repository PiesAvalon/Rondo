# `jax.scipy.linalg.solve` silently ignores `assume_a='sym'` / `'her'` (and `lower`)

**Component:** JAX 0.10.2 (CPU), `jax.scipy.linalg.solve` -> `jax._src.scipy.linalg._solve`.
**Severity:** silently wrong solution for any input whose two triangles differ.  The docstring
documents the SciPy contract - *"lower: Referenced only if `assume_a != 'gen'`. If True, only use the
lower triangle of the input, If False (default), only use the upper triangle"* and the option list
`"sym"` (symmetric) / `"her"` (hermitian) / `"pos"` - but the dispatch only special-cases `'pos'`:

```python
# jax/_src/scipy/linalg.py (master lines 1182-1185)
@jit(static_argnames=('assume_a', 'lower'))
def _solve(a, b, assume_a, lower):
  if assume_a != 'pos':
    return jnp_linalg.solve(a, b)          # <- 'sym' and 'her' fall back to the GENERAL solve
  ...                                      # only 'pos' uses the cholesky path with `lower`
```

so `assume_a='sym'`/`'her'` are validated (invalid values raise) but have no effect, `lower` is
ignored with them, and the answer is the generic LU solution - neither SciPy triangle convention:

```python
A = [[2., 1.], [0., 3.]], b = [1., 2.]
scipy.linalg.solve(A, b, assume_a='sym', lower=False)   # [0.2, 0.6]        (upper triangle)
scipy.linalg.solve(A, b, assume_a='sym', lower=True )   # [0.5, 0.666667]   (lower triangle)
jax.scipy.linalg.solve(A, b, assume_a='sym', lower=False)   # [0.166667, 0.666667]
jax.scipy.linalg.solve(A, b, assume_a='sym', lower=True )   # [0.166667, 0.666667]  identical
```

**Status:** present in 0.10.2 and at master (`jax/_src/scipy/linalg.py:1184`), in eager mode and under
`jax.jit`.  `assume_a='pos'` *does* honour `lower` (its cholesky path is used), and the neighbouring
`solve_triangular(lower=)`, `lu_solve(trans=)` and `eigh_tridiagonal(select=)` honour their selectors.
**Found with:** the wave-376 continuation of the wave-373 documented-selector sweep over
`jax.scipy.linalg`; certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 34 checks).

---

## Summary

| witness | `jax` (`sym`/`her`) | `scipy` `lower=True` | `scipy` `lower=False` |
|---|---|---|---|
| `[[2,1],[0,3]]`, `b=[1,2]` (`sym`) | `[0.166667, 0.666667]` | `[0.5, 0.666667]` | `[0.2, 0.6]` |
| `[[4,1,.5],[2,5,1],[.5,3,6]]`, `b=[1,2,3]` (`sym`) | `[0.137405, 0.274809, 0.351145]` | `[0.162162, 0.061776, 0.455598]` | `[0.123007, 0.287016, 0.441913]` |
| `[[2,1+1j],[0,3]]`, `b=[1,2-1j]` (`her`) | `[0-0.166667j, 0.666667-0.333333j]` | `[0.5+0j, 0.666667-0.333333j]` | `[0-0.25j, 0.75-0.25j]` |

In every case the JAX answer is bit-identical for `lower=True` and `lower=False` (to 1e-12) and equal
to `assume_a='gen'`, i.e. the two parameters have no effect; SciPy's two settings differ from each
other and from the generic solve.  The same holds for matrix right-hand sides and batched `a` (the
flags are inert there too).

## Root cause

`_solve` branches on `assume_a != 'pos'` and returns `jnp_linalg.solve(a, b)` for everything else,
without looking at `assume_a` or `lower` again.  Only the `'pos'` branch builds the cholesky factor
with `cho_factor(..., lower=lower)`.  The public wrapper validates the four documented values
(`valid_assume_a = ['gen', 'sym', 'her', 'pos']`) before calling `_solve`, so `'sym'`/`'her'` are
accepted and silently degraded to the general path.

## Suggested fix

Either implement the symmetric/Hermitian branches (e.g. use the selected triangle via
`jnp.tril`/`jnp.triu` symmetrisation plus a symmetric solve, or `jax.scipy.linalg.ldl`/cholesky for
`'pos'`), or - if they are intentionally unsupported for now - **reject** `assume_a in ('sym', 'her')`
with a clear `NotImplementedError` (as `jax.scipy.linalg.eigh_tridiagonal(select='v')` already does)
and document it, instead of returning a different system's solution.  A regression test is the
non-symmetric 2x2 witness above, which must match `scipy.linalg.solve(A, b, assume_a='sym',
lower=...)` for both settings (or raise).

## Impact

* Porting SciPy code that relies on the documented triangle convention silently produces different
  solutions - the same failure shape as `jax.scipy.linalg.eigh`'s ignored `lower` (JAX-019), reached
  through the *linear-solve* entry point, which is the most-used `jax.scipy.linalg` function.
* The convention exists for inputs whose unused triangle is irrelevant or garbage (a triangular
  factor, a view into a larger buffer, a matrix assembled with one triangle filled in), and SciPy
  reads only the requested triangle.
* No error or warning is raised for a documented parameter combination, and the validation code
  that checks the option list makes the parameter look supported.

## Controls (all in `repro.py`, 34 checks, 0 failing on each of 2 replays)

* **Path control** - `assume_a='pos'` honours `lower` and matches SciPy exactly for both settings,
  so the triangle machinery exists in the same dispatch.
* **Symmetric/Hermitian control** - with a genuinely symmetric (resp. Hermitian) `a`, JAX's `'sym'`
  (resp. `'her'`) result equals SciPy's for both `lower` values, so the defect only shows when the
  triangles differ (the documented use of the parameter).
* **Generic control** - `assume_a='gen'` is triangle-independent in both implementations.
* **Shape control** - matrix right-hand sides and batched `a` behave the same way (the flags are
  inert), and the symmetric cases match SciPy.
* **Module control** - `jax.scipy.linalg.solve_triangular(lower=)`, `lu_solve(trans=0/1/2)` and
  `eigh_tridiagonal(select='i')` all honour their selectors and match SciPy.
* **Validation control** - an invalid `assume_a` still raises `ValueError`, i.e. the option list is
  checked even though `'sym'`/`'her'` are not implemented.
* **Lane control** - `jax.jit` returns the same ignored-flag solution as eager.

## Novelty

Searched the JAX tracker (issues and PRs) on 2026-09-13: `assume_a sym` and
`solve assume_a ignored` return **zero** hits; the four `assume_a` hits are all about
`assume_a='pos'` (jax#39986 batched `cho_solve` deprecation, jax#16149 NaNs) or unrelated
(jax#33081 `sici`, jax#13105 `lstsq`), and jax#1043 is the old `sym_pos=True` gradient request.  The
`if assume_a != 'pos'` line is still at master (line 1184), i.e. unfixed.  The campaign's JAX-019
covers `jax.scipy.linalg.eigh`'s ignored `lower` - a different function and code path; no JAX record
covers `solve`.

## Environment

```
jax 0.10.2 (CPU, x64 enabled), numpy 2.5.1, scipy 1.18.0, Python 3.12.13, WSL2
(the SciPy reference is imported in the same interpreter - see collect_env.txt and evidence.txt)
```

## Anticipated objections and responses

* **"`sym`/`her` are only performance hints."**  No - they change the problem that is solved
  (SciPy reads only the selected triangle), and the docstring says `lower` "is referenced only if
  `assume_a != 'gen'`", which is false for `'sym'`/`'her'` in JAX.  The returned vector differs from
  SciPy's by up to 3x on the witnesses.
* **"The input must be symmetric for `assume_a='sym'`."**  For a symmetric input there is no
  difference, which is exactly why the parameter exists for inputs where the unused triangle is
  irrelevant - the documented LAPACK/SciPy contract.  If JAX does not intend to support that, it
  should raise instead of silently solving a different system.
* **"Use `assume_a='pos'`."**  That only covers positive-definite matrices (and requires the
  cholesky path); `'sym'`/`'her'` are documented as the general symmetric/Hermitian options.
* **"This is the same as JAX-019."**  Same *shape* of defect (a documented selector ignored by a
  SciPy-compatibility wrapper) in a different function, different code path and different fix; the
  campaign keeps them as separate reports, and the priority order (solve before eigh for most users)
  makes this one independently worth fixing.

## Upstream context (wave-432 novelty re-sweep)

SciPy is deprecating the `lower`/`transposed` parameters of `linalg.solve` (scipy/scipy#23072).
JAX's wrapper still mirrors and documents the older SciPy signature, and the parameters are inert -
so the report stands, but a reviewer may note that the parameter set itself is being retired upstream.
