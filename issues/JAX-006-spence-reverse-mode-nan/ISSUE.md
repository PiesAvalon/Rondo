# `jax.scipy.special.spence` returns `NaN` reverse-mode gradients for `1 < x <= 2` while the forward value and forward mode are correct

**Component:** JAX 0.10.2, `jax.scipy.special.spence`; reverse mode (`jax.grad`, `jax.vjp`,
`jax.jacrev`), eager and `jax.jit`, float64 and float32.
Forward mode (`jax.jvp`), the forward value and the finite difference are all correct on the same
inputs.
**Severity:** silent wrong gradient (NaN) on a whole interval where the function is smooth and its
derivative is finite. An optimizer or a loss that touches `spence` at `1 < x <= 2` gets `NaN` from
`jax.grad` even though `jax.jvp` is fine, so the two autodiff modes disagree.
**Status:** present in JAX 0.10.2
**Found with:** a forward-mode-vs-reverse-mode consistency sweep of `jax.scipy.special` (the
`jvp`/`vjp` disagreement was the trigger), then a NaN-gradient scan over the branch boundaries of
31 special functions; the standalone tools ship next to this file.
**Upstream source:** `jax/_src/scipy/special.py`, `_spence_calc` (the `x < 0.5` branch,
`y_flag_one = pi**2/6 - log(x) * log(1 - x) - y`).

---

## Summary

`jax.scipy.special.spence(x)` differentiates incorrectly for `1 < x <= 2`:

| x | `spence(x)` (fp64) | `jax.grad(spence)(x)` | `jax.jvp(spence)(x)` | `-log(x)/(x-1)` | FD of `spence` |
|---|---|---|---|---|---|
| 1.0001 | `-0.0000999…` | `nan` | `-0.99995000333` | `-0.99995000333` | `-0.999950…` |
| 1.01   | `-0.0099003…` | `nan` | `-0.99503308532` | `-0.99503308532` | `-0.995033…` |
| 1.1    | `-0.0976052352293` | `nan` | `-0.95310179804` | `-0.95310179804` | `-0.953101…` |
| 1.25   | `-0.235900297686`  | `nan` | `-0.89257420526` | `-0.89257420526` | `-0.892574…` |
| 1.5    | `-0.448414206924`  | `nan` | `-0.81093021622` | `-0.81093021622` | `-0.810930…` |
| 1.75   | `-0.64276126884`   | `nan` | `-0.74615438389` | `-0.74615438389` | `-0.746154…` |
| 2.0    | `-0.822467033424`  | `nan` | `-0.69314718056` | `-0.69314718056` | `-0.693147…` |
| 2.01   | `-0.828788198783`  | `-0.6912224971` | `-0.6912224971` | `-0.6912224971` | `-0.691222…` |

`spence` is the dilogarithm (`Li_2`-family) and is smooth on `(0, 2)`; `spence(1.5)` from JAX is
bit-identical to SciPy 1.18.0 (`-0.4484142069236462`), and the central difference of the JAX forward
matches `-log(x)/(x-1)`.  Reverse mode is the only lane that fails, and it fails for **every** input
in the interval, in eager and `jit`, in float32 and float64:

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy import special as jsp

float(jax.grad(lambda v: jsp.spence(v))(jnp.float64(1.5)))   # nan
float(jax.jvp(lambda v: jsp.spence(v), (jnp.float64(1.5),), (jnp.float64(1.0),))[1])  # -0.8109302162...
float(jax.jit(jax.grad(lambda v: jsp.spence(v)))(jnp.float64(1.5)))  # nan
```

`jax.jacrev` and `jax.vjp` reproduce it (`jax.vjp(spence, 2.0)[1](1.0) == nan`), so it is the
reverse-mode rule rather than the `grad` wrapper.  Outside the interval reverse mode is exact
(`x = 0.1, 0.5, 0.9, 0.99, 2.01, 2.5, 3.0, 4.0` all match the analytic derivative to `<= 1e-12`).

The scan over 31 `jax.scipy.special` functions (grid of branch boundaries, eager and `jit`, NaN
gradient with finite forward and finite FD) reports `spence` as the **only** hit.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(73 checks per replay, 2 fresh processes.)

What the reproducer pins:

| dimension | result |
|---|---|
| trigger cells | seven points in `(1, 2]` (1.0001, 1.01, 1.1, 1.25, 1.5, 1.75, 2.0): the forward is finite and equals SciPy, `jax.grad` is `nan`, `jax.jvp` equals the analytic derivative, and the FD of the forward equals it too |
| modes | `jax.grad`, `jax.jit(jax.grad)`, `jax.vjp`, `jax.jacrev` all produce `nan`; `jax.jvp` produces the correct value on every cell |
| dtypes | float64 and float32 (`float32 grad spence(2.0) = nan`) |
| controls | eight cells outside the interval (0.1, 0.5, 0.9, 0.99, 2.01, 2.5, 3.0, 4.0) match the analytic derivative to `<= 1e-9` |
| mechanism | the minimal witness `jnp.where(v < 0.5, jnp.log(v) * jnp.log(1 - v), v)` at `v = 1.5` has value `1.5`, **`nan` reverse-mode gradient** and forward-mode gradient `1.0`; the additive form (`log(v) + log(1-v)`) keeps a finite gradient, which isolates the failure to the **product** vjp (`0 * NaN`) |
| secondary | at exactly `x = 1.0` both modes return `0.0` where the derivative is `-1.0` (the value branch special-cases `x == 1 -> 0`) |

## Root cause (pointer)

`_spence_calc` computes its `x < 0.5` branch unconditionally:

```python
y_flag_one = np.pi ** 2 / 6.0 - jnp.log(x) * jnp.log(1.0 - x) - y   # NaN for x > 1
y = jnp.where(x_5_bool, y_flag_one, y)                              # the value is masked ...
```

For `1 < x <= 2`, `1 - x < 0`, so `log(1 - x)` is `NaN` even though that branch is not selected.
The forward is masked and stays finite, but the reverse-mode cotangent flowing into the product
`log(x) * log(1 - x)` is zero, and the vjp of a product computes `0 * log(1 - x) = 0 * NaN = NaN`.
Forward mode is unaffected because a zero tangent propagates through the product as zero - hence
`jvp` correct, `vjp` `NaN`.  `lax.select`/`jnp.where` cannot mask this on its own; the branch value
itself has to stay finite.  A local fix is to make the dead value finite before the mask, e.g.

```python
one_minus_x = jnp.where(x < 0.5, 1.0 - x, 0.5)      # only the selected branch is used
y_flag_one = np.pi ** 2 / 6.0 - jnp.log(x) * jnp.log(one_minus_x) - y
```

(or to guard the `log` argument the same way elsewhere in the function).  A regression test is
`jax.grad(spence)(x) == -log(x)/(x-1)` for `x` in `{1.001, 1.1, 1.5, 2.0}`.

## Novelty

* `gh search issues --repo jax-ml/jax` for `spence`, `spence grad`, `dilogarithm gradient` returns
  only [#27088](https://github.com/jax-ml/jax/issues/27088) (the `jax.scipy.special` umbrella, whose
  `spence` row reads ``tested on R+, JIT`` and lists no defect) and
  [#13239](https://github.com/jax-ml/jax/pull/13239)-adjacent history (the PR that *added* `spence`).
* The campaign's JAX drafts cover other functions (`expn`, `poch`, `isf`, `export`, fused `power`);
  none of them is `spence`, and the JAX-138 boundary pass did not include it.
* The documented limitation in the JAX docstring ("unlike SciPy, only defined for positive real
  values; NaN for negative values") is a different, documented behaviour and is not this report.
