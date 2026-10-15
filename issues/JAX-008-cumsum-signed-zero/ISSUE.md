# JAX prefix scans drop the sign of IEEE `-0.0`: `cumsum`, `cumprod` and `lax.associative_scan`

**Component:** JAX 0.10.2, CPU (XLA): `jax.numpy.cumsum`, `jax.numpy.cumprod` and
`jax.lax.cumsum` / `jax.lax.associative_scan` with `add`, `multiply`, `maximum` and `minimum`, on
`float32`/`float64` and `complex64`/`complex128`
**Severity:** silent wrong sign of zero — the first output element is not even a copy of the input,
and downstream `signbit`-sensitive code (`1/x`, branch cuts, `copysign`) takes the opposite branch
**Status:** present in JAX 0.10.2; NumPy (and MindSpore) preserve the sign
**Found with:** the wave-181 complex/real reduction grid (Reference vs Fresh oracle: eager vs
`jax.jit`), minimised to the sign-bit witnesses below; certified reproducer `repro.py` in this
directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

`cumsum` is a prefix sum, so its **first** output element is the input itself, and every later
element is an IEEE addition, which keeps the sign of a zero.  JAX returns positive zeros:

```python
import numpy as np, jax, jax.numpy as jnp
x = jnp.asarray([-0.0, -0.0])
np.cumsum(np.asarray(x))        # [-0.0, -0.0]       (sign bits [1, 1])
jnp.cumsum(x)                   # [ 0.0,  0.0]       (sign bits [0, 0])  <- element 0 is already wrong
jax.jit(jnp.cumsum)(x)          # [ 0.0,  0.0]
```

| input | NumPy (sign bits) | `jnp.cumsum` eager | `jax.jit` | `lax.cumsum` / `associative_scan` |
|---|---|---|---|---|
| `[-0.0, -0.0]` | `[1, 1]` | `[0, 0]` | `[0, 0]` | `[0, 0]` |
| `[-0.0, -0.0, -0.0]` | `[1, 1, 1]` | `[0, 0, 0]` | `[0, 0, 0]` | - |
| `[-0.0, 0.0, -0.0]` | `[1, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0]` | - |
| `[-0.0, 1.0, -0.0]` | `[1, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0]` | - |
| `float32 [-0.0, -0.0]` | `[1, 1]` | `[0, 0]` | `[0, 0]` | - |
| complex `[-0-0j, -0-0j]` | `[(1,1), (1,1)]` | `[(0,0), (0,0)]` | `[(0,0), (0,0)]` | - |
| complex `[+0-0j, +0-0j]` | `[(0,1), (0,1)]` | `[(0,0), (0,0)]` | `[(0,0), (0,0)]` | - |

A single-element input *is* preserved by `jnp.cumsum` (`[-0.0] -> [-0.0]`), while
`lax.cumsum`/`associative_scan` lose the sign for two elements - so the loss is in the cumulative
combine, not in the input handling.

## Scope and controls

* `jnp.cumsum` of non-zero arrays (`[1, 2, 3]`, `[1.5, -2.5, 0.25]`, all-zero `[0, 0, 0]`) matches
  NumPy exactly, and `[0.0, -0.0]` -> `[0.0, 0.0]` is right (opposite-sign zeros add to `+0.0`).
* A single-element `jnp.cumsum([-0.0])` keeps the sign, so the defect needs an actual combine.
* `jax.lax.cumsum` and `jax.lax.associative_scan(jnp.add, ...)` show the same loss, i.e. it is a
  lower-level XLA scan issue rather than a NumPy-API shim.

## The product scan shows the same identity loss for complex inputs

`jnp.cumprod` is correct for real inputs (all eight real zero-sign cases match NumPy), but for
complex inputs the **identity element** loses the real sign:

| input | NumPy | `jnp.cumprod` | `jax.jit` | `lax.associative_scan(jnp.multiply)` |
|---|---|---|---|---|
| `[(-0-0j)]` | `[(1,1)]` | `[(0,1)]` | `[(0,1)]` | `[(1,1)]` (single element: no combine) |
| `[(-0-0j), (-0-0j)]` | `[(1,1), (0,0)]` | `[(0,1), (1,0)]` | - | `[(0,0), ...]` |

The single-element `cumprod` is the identity, so `[(0,1)]` instead of `[(1,1)]` is a defect of the
same class as the `cumsum` one; `associative_scan(jnp.multiply, ...)` on two elements loses both
signs, while a single element is preserved.  NumPy is the only implementation that returns the input
unchanged.

## The max/min scans contradict JAX's own `maximum`/`minimum`

For `[-0.0, -0.0]`, JAX's own pointwise and reduction operators preserve the sign, while the scan
does not:

| call | sign bits |
|---|---|
| `jnp.maximum(-0.0, -0.0)` | `[1]` (`-0.0`) |
| `jnp.minimum(-0.0, -0.0)` | `[1]` |
| `jnp.max([-0.0, -0.0])` / `jnp.min(...)` | `[1]` / `[1]` |
| NumPy `maximum.accumulate` / `minimum.accumulate` | `[1, 1]` / `[1, 1]` |
| **`lax.associative_scan(jnp.maximum, ...)`** | **`[0, 0]`** |
| **`lax.associative_scan(jnp.minimum, ...)`** | **`[0, 0]`** |

and for `[-0.0, -1.0]` the scan's **first** result is `+0.0` although `maximum(-0.0, -1.0) = -0.0`.
This is independent of JAX's tie-breaking convention for `+0.0` vs `-0.0` (which differs from NumPy
for `[0.0, -0.0]` and is recorded as a convention): the sign is lost even where every JAX operator
that is not a scan returns it, so the scan lowering is the inconsistent component.

## Impact

A cumulative sum of data that contains negative zeros (a difference of equal values, a product of
zeros, an explicitly padded `-0.0`) loses the sign in every prefix; code that then inspects
`signbit` - or computes `1/prefix`, `angle`, or a branch-cut-sensitive function of it - silently
takes the wrong branch, and the error starts at the very first element rather than accumulating.

## Novelty

Searched the JAX tracker for `cumsum negative zero`, `cumsum signed zero`, `cumsum -0.0` and
`scan negative zero sign`: no report exists (the hits for `-0.0` are unrelated memory/PRNG issues).
PyTorch has the same class of defect in `torch.cumsum` (pytorch#195851, open), but that is a
different project and does not cover JAX; the campaign's other JAX drafts (JAX-001…007) are other
functions and paths, and its signed-zero work for other frameworks (`PT-012`, `PT-019`, `PT-020`,
`PT-021`, `PT-024`, `TF-003`, `TF-007`) does not mention `cumsum`.

## Environment

JAX 0.10.2, NumPy 2.5.1, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64, RTX 4090 (CUDA 13.0); all
checks ran on the CPU (XLA:CPU).  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"`-0.0` inputs are a corner case."**  The first output element being wrong shows this is not an
  accumulation-order subtlety: `cumsum` must return the input element unchanged there, and NumPy and
  MindSpore both do.
* **"The sign of a zero is unobservable."**  `signbit`, `1/x`, `angle` and branch-cut functions all
  read it; JAX itself exposes `jnp.signbit` and preserves signed zeros through `add`.
