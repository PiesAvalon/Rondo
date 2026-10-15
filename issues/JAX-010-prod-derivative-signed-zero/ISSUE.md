# The derivative of `jnp.prod` loses the sign of IEEE zeros (`grad`, `vjp`, `jacrev`, `jacfwd`, `jvp`)

**Component:** JAX 0.10.2 / jaxlib 0.10.2, CPU: the `reduce_prod` derivative
(`jax.grad`/`jax.vjp`/`jax.jacrev` and the forward-mode `jax.jvp`/`jax.jacfwd` of `jnp.prod`,
`jnp.prod(axis=...)` and `lax.reduce_prod`); `float32` and `float64`.
**Severity:** silent wrong sign of zero in a gradient.  The magnitudes are correct and only the sign
bits differ, but a sign of zero is observable (`1/grad` becomes `+inf` where the exact derivative
gives `-inf`, `signbit`/`copysign` flip), and the returned bits are not even self-consistent: reverse
and forward mode disagree with each other on the same input, in both directions.
**Status:** present in JAX 0.10.2 on CPU.  JAX's own forward primitives (`jnp.prod`,
`lax.reduce_prod`, `jnp.cumprod`, `lax.mul`) preserve the sign, so the derivative is the outlier;
PyTorch's `torch.func.jacrev` and `torch.compile`d backward return the exact signed partials for the
same problem (this campaign's PT-027).
**Found with:** the wave-204/205 Rondo prod-gradient signed-zero oracle (`jax.grad`/`vjp`/`jacrev`
vs per-basis `jax.jvp` vs the exact float64 product of the other factors); certified reproducer
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 260 checks).

---

## Summary

`d prod / d x_i = prod_{j != i} x_j`; IEEE-754 fixes the sign of every zero factor product, so the
gradient of a product containing zeros has well-defined `-0.0` entries.  JAX returns positive zeros,
already with a **single** zero factor:

```python
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)

x = jnp.asarray([1.0, -0.0])
jnp.prod(x)                              # -0.0                     <- the value keeps the sign
jax.grad(lambda t: jnp.prod(t))(x)       # [+0.0, 1.0]              <- sign lost
np.array([-0.0, 1.0])                    # exact partials
jnp.cumprod(x)                           # [1.0, -0.0]              <- cumprod keeps it too

x = jnp.asarray([-0.0, -0.0])
jax.grad(lambda t: jnp.prod(t))(x)                     # [+0.0, +0.0]
jax.jacfwd(lambda t: jnp.prod(t))(x)                   # [-0.0, -0.0]
[jax.jvp(lambda t: jnp.prod(t), (x,), (e,))[1] for e in
 (jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0]))]   # [-0.0, -0.0]
```

Measured (float64 and float32 behave identically; `exact` is the float64 product of the other
factors):

| input | exact partials | `grad`/`vjp`/`jacrev` | `jacfwd`/per-basis `jvp` |
|---|---|---|---|
| `[-0.0, -0.0]` | `[-0.0, -0.0]` | `[+0.0, +0.0]` | `[-0.0, -0.0]` (right) |
| `[1.0, -0.0]` | `[-0.0, 1.0]` | `[+0.0, 1.0]` | `[+0.0, 1.0]` |
| `[-0.0, 2.0, 3.0]` | `[6.0, -0.0, -0.0]` | `[6.0, +0.0, +0.0]` | `[6.0, +0.0, +0.0]` |
| `[2.0, -0.0, 3.0]` | `[-0.0, 6.0, -0.0]` | `[+0.0, 6.0, +0.0]` | `[+0.0, 6.0, +0.0]` |
| `[-0.0, 0.0, 0.8]` | `[+0.0, -0.0, -0.0]` | `[+0.0, +0.0, +0.0]` | `[+0.0, +0.0, +0.0]` |
| `[1.0, -0.0, -0.0, 1.0]` | `[+0.0, -0.0, -0.0, +0.0]` | `[+0.0] * 4` | `[-0.0] * 4` |
| no zeros (control) | `[-2.4, -1.2, 0.8, -6.0]` | exact | exact |
| all `+0.0` (control) | `[+0.0, +0.0]` | exact | exact |
| `[-0.0]` (control) | `[1.0]` | exact | exact |

Non-zero magnitudes match the exact partials in every lane and both controls agree, so the whole
divergence is the sign of zero.  `jax.jit(jax.grad(...))` and `jnp.prod(axis=0)` behave exactly like
`jax.grad`.  Note the two all-negative-zero rows: there reverse mode is wrong while forward mode is
right, and for `[1.0, -0.0, -0.0, 1.0]` reverse mode returns `+0.0` where forward mode returns
`-0.0` and the exact value is `+0.0` at index 0 and `-0.0` at index 1 - i.e. each lane is wrong in a
different direction.

## Root cause

`jnp.prod` is `lax.reduce_prod`, whose derivative is the generic reduction JVP
(`jax/_src/lax/lax.py`):

```python
def _reduce_prod_jvp_rule(primals, tangents, *, axes):
  reducer = lambda x, y: [mul(x, y)]
  primals_out, tangents_out = _reduce_jvp(reducer, [_const(primals[0], 1)],
                                          primals, tangents, axes)
  return primals_out[0], tangents_out[0]
```

`_reduce_jvp` reduces the operands pairwise into a product tree and then differentiates that tree
with `api.jvp`, i.e. each node contributes `tangent_x * y + x * tangent_y`.  The directional
derivative is therefore evaluated as a **sum of products**, and `(+0.0) + (-0.0) = +0.0` in
round-to-nearest, so a `-0.0` partial is turned into `+0.0` (or, when the terms arrive in the other
order, a `+0.0` partial is turned into `-0.0`, as in the four-element row).  Reverse mode is the
transpose of the same rule, so it inherits the signs of that evaluation; the tree shape and the
pairing decide the result, which is why forward and reverse mode can disagree.  The primitives
themselves are exact: `lax.mul(1.0, -0.0)` is `-0.0` and `jnp.prod([1.0, -0.0])` is `-0.0`.

## Impact

* `1/grad` at such an entry is `+inf` with JAX and `-inf` with the exact derivative (the reproducer
  asserts both), and `signbit`-based logic (`copysign`, sign masks, branch selection, comparisons
  against `-0.0`) takes the opposite branch.
* The framework is internally inconsistent: the product's *value*, `cumprod` and `mul` keep the
  sign, and forward and reverse differentiation of the same function return **different** gradient
  bits, so a user cannot even work around it by choosing a mode.
* Products of scale parameters, products of gates and any Jacobian analysis built on `prod` get
  gradients of the wrong sign at exactly the zero entries; magnitudes look right, so tolerance-based
  gradient checks pass.

## Suggested fix

Compute the zero entries of a product's derivative with the sign-carrying product of the other
factors rather than by summing `tangent_x * y + x * tangent_y` terms, e.g. via exclusive prefix and
suffix `cumprod` products (the same formula the ATen decomposition of `prod`'s backward uses, and
what `torch.func.jacrev` returns for the same input): `grad_i = prod_{j < i} x_j * prod_{j > i} x_j`.
A regression test needs no reference library:

```python
x = jnp.asarray([1.0, -0.0])
g = jax.grad(lambda t: jnp.prod(t))(x)
assert np.signbit(np.asarray(g)[0])          # currently False; the exact partial is -0.0
```

## Controls (all in `repro.py`, 260 checks, 0 failing on each of 2 replays)

* The exact reference is the float64 product of the other factors; the no-zeros, all-`+0.0` and
  single-element cases match it in every lane, so only negative-zero signs diverge.
* Non-zero magnitudes are checked to agree with the exact partials in both mode groups, so no claim
  rests on magnitude errors.
* For every lane and case, the differing entries are checked to be zero-sign flips (both the JAX and
  the exact value are `±0.0` in those positions).
* The forward primitives are checked directly: `jnp.prod`, `lax.reduce_prod`, `jnp.cumprod` and
  `lax.mul` all keep `-0.0` on the same input, which localises the defect to the derivative rule.
* `jax.jit(jax.grad(...))`, `jax.vjp`, `jax.jacrev` and `jnp.prod(axis=0)` are checked to behave
  like `jax.grad`; `jax.jacfwd` and the per-basis `jax.jvp` are checked to equal each other; both
  widths are covered.

## Novelty

`gh search issues --repo jax-ml/jax` for `prod signed zero`, `prod gradient -0.0`, `reduce_prod
zero`, `jnp.prod gradient`, `prod jvp`, `prod transpose`, `prod zeros gradient`, `negative zero`,
`sign of zero` and `signbit` returns no report about the derivative of `prod` losing zero signs.
The adjacent items are [#10206](https://github.com/jax-ml/jax/issues/10206) (gradient of `jnp.prod`
on a **complex** three-element array when jitted, closed 2022), [#672](https://github.com/jax-ml/jax/issues/672)
(cannot take the gradient of `np.prod`, ancient), [#1888](https://github.com/jax-ml/jax/issues/1888)
(grad of `prod` of `take` with empty indices, closed) and
[#2380](https://github.com/jax-ml/jax/issues/2380) (derivative of the determinant of a singular
matrix, closed) - none of them is about the sign of zero, and none observes the forward/reverse
disagreement.  The campaign's JAX-008 covers prefix scans (`cumsum`/`cumprod`) dropping `-0.0`, a
different primitive and code path, and PT-027 is the same class for PyTorch's `prod` backward (where
the failing region is inputs with **more than one** zero and the cause is an `at::zeros_like`
shortcut; here a single zero suffices and the cause is the tree-multiply evaluation order).

## Environment

JAX 0.10.2 / jaxlib 0.10.2, CPU backend, Python 3.12.13, NumPy 2.5.3, WSL2 (Linux
6.6.87.2) x86_64, `jax_enable_x64=True`.  Full dump in `collect_env.txt`; transcripts in
`evidence.txt`.

---

## Anticipated objections and responses

* **"The derivative really is zero, so the sign is cosmetic."**  The value is zero but not signless:
  IEEE-754 fixes it, `1/grad` returns the opposite infinity, `signbit` flips, and JAX itself returns
  the correct signed zeros through `jnp.prod`, `lax.reduce_prod`, `jnp.cumprod` and `lax.mul` on the
  same input - and forward and reverse differentiation disagree with each other, which no
  "the derivative is zero" argument explains.
* **"A sum of products is a legitimate evaluation of the JVP."**  For a general direction that is
  true, but `jax.grad` computes the full gradient (equivalently the transposed JVP), and for the
  gradient the exact partials are well defined.  `jnp.jvp` with a *basis* direction `e_i` is also the
  partial itself; the reproducer checks that too.  A sign-exact evaluation exists and is used by
  other implementations of the same op.
* **"Numerical gradient checks pass."**  They do - magnitudes are correct - and the reproducer says
  so explicitly.  The report is deliberately narrow: signed zeros and lane self-consistency.
* **"Duplicates PT-027 / JAX-008."**  PT-027 is PyTorch's no-`dim` `prod` backward (a `zeros_like`
  shortcut for inputs with more than one zero); this is JAX's reduction JVP, which fails with a
  single zero and shows forward/reverse disagreement.  JAX-008 is `cumsum`/`cumprod` prefix scans.
  Different frameworks, primitives and mechanisms.
