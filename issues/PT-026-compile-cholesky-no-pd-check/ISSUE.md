# `torch.compile` drops `torch.linalg.cholesky`'s positive-definiteness check and returns the unchecked `cholesky_ex` factor

**Component:** PyTorch 2.12.1+cu130, **CPU**, eager vs `torch.compile` (Inductor), `float32` and
`float64`.
**Severity:** silent wrong values plus a lost error signal - for a non-positive-definite matrix eager
raises `torch._C._LinAlgError`, while the compiled call returns a **finite, plausible-looking**
lower-triangular matrix whose diagonal can be negative and whose product is not the input (off by 12
on the 3x3 witness).  The exception that libraries use to detect non-PD matrices disappears only when
the graph is compiled.
**Status:** present in 2.12.1+cu130 (CPU); `Tensor.cholesky()` still raises under `torch.compile`, and
positive-definite inputs are byte-identical between the lanes.
**Found with:** the wave-193 Rondo linalg edge-matrix grid (16 `torch.linalg` ops x 8 degenerate
matrices x 2 dtypes, Reference = eager, Fresh = `torch.compile` with a cold Inductor cache), with the
`info` channel of `torch.linalg.cholesky_ex` as an independent reference and the generated Inductor
source as the mechanism; certified reproducer is `repro.py` in this directory (`--replays 2` ->
`RESULT: 0 failing replay(s).`, 42 checks).

---

## Summary

`torch.linalg.cholesky` documents that it raises when the factorization cannot be computed, and eager
does.  Under `torch.compile` the same call returns the raw LAPACK result instead:

```python
import torch
A = torch.tensor([[1., 2., 0.],
                  [2., 1., 0.],
                  [0., 0., 1.]])          # eigenvalues -1, 1, 3 -> not positive definite

torch.linalg.cholesky(A)
# torch._C._LinAlgError: linalg.cholesky: The factorization could not be completed because the
# input is not positive-definite (the leading minor of order 2 is not positive-definite)

L = torch.compile(torch.linalg.cholesky)(A)
# tensor([[ 1., 0., 0.],
#         [ 2., -3., 0.],
#         [ 0.,  0., 1.]])              # no exception, negative diagonal

(L @ L.T)
# tensor([[ 1.,  2., 0.],
#         [ 2., 13., 0.],               # != A (the [1][1] entry is 13 instead of 1)
#         [ 0.,  0., 1.]])
```

The same happens for other degenerate inputs, and the compiled backward pass happily differentiates
the bogus factor:

| input | eager | `torch.compile` |
|---|---|---|
| non-PD (f32/f64) | raises `_LinAlgError` | finite factor, `diag = [1, -3, 1]`, `max|L@L.T - A| = 12` |
| zero matrix | raises `_LinAlgError` | all-zeros factor (`L@L.T = 0 != A`) |
| singular matrix (f64) | raises `_LinAlgError` | finite factor `[[1,0,0],[2,0,0],[1,1,1]]` |
| matrix with `nan` | raises `_LinAlgError` | factor with `nan` on the diagonal |
| matrix with `inf` | raises `_LinAlgError` | factor with `inf`/`nan` entries |
| `cholesky(A).sum().backward()` | raises `_LinAlgError` | returns a gradient (f32) |
| positive-definite input | factor | identical factor (control) |
| `Tensor.cholesky()` | raises | still raises (control) |

`torch.linalg.cholesky_ex` returns exactly the same factor **and** an `info` tensor (2 for the
non-PD witness) in both lanes, so the information needed to raise is still produced - it is the check
that is missing.

## Root cause

Inductor decomposes `aten.linalg_cholesky` to the `_ex` variant, whose `check_errors` argument
defaults to `False`.  The generated kernel for the witness contains:

```python
# Topologically Sorted Source Nodes: [linalg_cholesky], Original ATen: [aten.linalg_cholesky_ex]
buf0 = torch.ops.aten.linalg_cholesky_ex.default(arg0_1)
```

The eager `aten.linalg_cholesky` op performs the `info != 0` -> `torch._C._LinAlgError` validation;
the lowered `linalg_cholesky_ex.default` call does not, and the `info` tensor is dropped from the
graph (only `buf0`, the factor, survives).  The result is a compiled `cholesky` that is exactly the
unchecked `cholesky_ex` factor.

## Impact

* The common "is this matrix positive definite?" idiom - call `torch.linalg.cholesky` inside
  `try/except torch._C._LinAlgError` (or `RuntimeError`) - silently stops working after compilation:
  the code takes the success branch and proceeds with a factor that does not satisfy `L @ L.T = A`.
* Unlike the `inv`/`solve` cases, whose compiled results are visibly NaN-filled, the Cholesky factor
  here is **finite and structurally plausible** (lower-triangular, unit-ish diagonal scale), so it is
  easy to consume silently in a log-determinant, whitening or Gaussian-process path.
* A compiled backward pass produces gradients from that invalid factor instead of raising, so the
  failure can also corrupt training rather than stopping it.

## Propagation through a public API: `MultivariateNormal` on a non-PD covariance

The dropped check is not confined to the low-level call.  `torch.distributions.MultivariateNormal`
computes its scale factor with `torch.linalg.cholesky`, so a non-positive-definite covariance matrix
behaves differently in the two lanes (measured in a fresh process with an empty
`TORCHINDUCTOR_CACHE_DIR`, `mvn_witness.py`):

| expression | eager | `torch.compile` |
|---|---|---|
| `MultivariateNormal(zeros(2), covariance_matrix=[[1,2],[2,1]]).log_prob(ones(2))` | raises `ValueError` | returns `nan` |
| `... .sample()` | raises `_LinAlgError` | **returns a sample** (`-0.19, 0.06` class of values) |
| `... .entropy()` | raises `_LinAlgError` | returns `nan` |
| the same with a PD covariance (control) | value | identical value |

The `sample()` row is the sharpest: the compiled path not only skips the error, it **draws data from
a distribution whose covariance is not positive definite**, using the bogus factor shown above.
Inside a single process the outcome can still be an exception when Inductor falls back to the eager
op for the internal `cholesky` call, which is why the reproducer runs this witness in its own cold
process (the same warm-cache fallback effect that PT-025 documented).

## Suggested fix

Keep the validation in the lowered form, for example by lowering `aten.linalg_cholesky` to a variant
that carries `check_errors=True`, or by emitting an `info != 0` check (raising
`torch._C._LinAlgError`) after `linalg_cholesky_ex` in the decomposition.  A `torch.compile` test
that asserts the exception for a non-PD input (and equality with eager for a PD one) would pin the
behaviour.  The same question applies to `linalg_inv`/`linalg_solve`/`linalg_eigvalsh`, whose eager
error checks are also missing from the compiled path (see the secondary witnesses in `repro.py`).

## Controls (all in `repro.py`, 50 checks)

* Positive-definite input: identical factor in both lanes, and `L @ L.T == A` within 1e-5.
* `MultivariateNormal` with a positive-definite covariance gives the same `log_prob` in both lanes,
  while the non-PD covariance witness (cold cache) shows the impact above.
* `Tensor.cholesky()` still raises in both lanes on the non-PD matrix, so the regression is in the
  functional lowering, not in Cholesky as such.
* `torch.linalg.cholesky_ex` (both `upper=False` and `upper=True`) agrees between the lanes on the
  same matrix - it is the documented non-raising variant, and the compiled `cholesky` equals its
  output exactly.
* `torch.linalg.cholesky_ex(..., check_errors=True)` **raises `_LinAlgError` in both lanes** on the
  same non-PD matrix, so the compiled world can perform the check when it is asked for explicitly;
  only the decomposition of the plain `cholesky` call drops it.  `linalg.inv_ex`/`linalg.solve_ex`
  behave the same way (their `check_errors=True` still raises under `torch.compile`).
* The compiled factor is checked directly: negative diagonal and `max|L@L.T - A| = 12` for the
  non-PD witness, all-zeros for the zero matrix, and non-finite entries for the `nan`/`inf` inputs.
* Secondary class (same missing check, visibly-NaN results): `torch.linalg.inv` on a singular matrix
  and `torch.linalg.eigvalsh` on an `inf` entry both raise eagerly and return NaN-filled tensors
  under `torch.compile`.

## Novelty

Searched the PyTorch tracker for `torch.compile cholesky not positive definite`,
`inductor linalg cholesky raise`, `linalg_cholesky_ex check_errors inductor` and
`torch.compile missing _LinAlgError`: no report matches.  The adjacent inductor-linalg reports are
[#151309](https://github.com/pytorch/pytorch/issues/151309) (type checking of `torch.linalg.inv` on
inductor), [#194005](https://github.com/pytorch/pytorch/issues/194005) (graphs containing
`linalg.cholesky`/`inv`/`solve`/`lu_factor` are never cached) and
[#194499](https://github.com/pytorch/pytorch/pull/194499) (caching for linalg ops with effects) -
about validation of a different op and about caching, not about the dropped positive-definiteness
check.  The campaign's other PyTorch drafts are unrelated (PT-022 unary zero flush, PT-023
`set_default_dtype`, PT-024 complex add, PT-025 float8 `abs`).

## Environment

PyTorch 2.12.1+cu130 (CPU execution), NumPy 2.5.3, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64, CPU
capability `AVX2`, Inductor C++ codegen.  Full dump in `collect_env.txt`; transcripts in
`evidence.txt`.

---

## Anticipated objections and responses

* **"Compiled linalg ops fall back to eager, so they must raise."**  They do not fall back here: the
  generated code calls `torch.ops.aten.linalg_cholesky_ex.default` inside the compiled graph, which
  is why the check is gone.  `Tensor.cholesky()` does still raise, which shows the difference is the
  decomposition of the functional op.
* **"The matrix is invalid input."**  A non-positive-definite matrix is an ordinary runtime
  condition that the API is *documented* to report, and eager does report it; upstream projects rely
  on that.  The bug is that compiling the same program silently changes the contract.
* **"NaN-filled results are equally easy to spot."**  For `inv`/`eigvalsh` they are, which is why the
  draft separates them; the Cholesky factor is finite and plausible (`[[1,0,0],[2,-3,0],[0,0,1]]`),
  so the wrong value can propagate without any NaN warning at all.
* **"Only `check_errors` defaults differ."**  Exactly - and that default is the bug for this call
  path: the eager op's contract includes the check, so the decomposition must preserve it.
