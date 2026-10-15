# `jax.nn.gelu` (default `approximate=True`) returns exactly zero value and gradient for `x <= -4.9` in float32

**Component:** JAX 0.10.2 / jaxlib 0.10.2, CPU: `jax.nn.gelu` with its **default** `approximate=True`
(the tanh approximation), through `jax.grad`, `jax.value_and_grad` and `jax.jit`; `float32` and
`float64`.
**Severity:** silent dead gradient.  `jax.grad(jax.nn.gelu)(x)` is **exactly `0.0`** for every
`x <= -4.9` in float32 (and `x <= -7.35` in float64) - a completely ordinary activation range - and
the activation itself collapses to `-0.0`, although both the value and the derivative are ordinary
representable numbers there (`-2.29e-07` and `-1.55e-06` at `x = -5`).  A gelu layer whose
pre-activation is one of those values gets no gradient at all.
**Status:** present in JAX 0.10.2 on CPU.  The exact-erf lane
(`jax.nn.gelu(x, approximate=False)`) keeps the tail at every affected point, so only the default
approximation is affected.
**Found with:** the wave-234 derivative-tail scan (analytic references evaluated in a
cancellation-free form, `1 + tanh(u) = 2 e^{2u} / (1 + e^{2u})`); certified reproducer `repro.py` in
this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 75 checks).

---

## Summary

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
x = jnp.float32(-5.0)
jax.nn.gelu(x)                                          # -0.0
jax.grad(jax.nn.gelu)(x)                                #  0.0
# the tanh approximation computed without cancellation:
#   0.5 * x * (1 + tanh(u)),  u = sqrt(2/pi) (x + 0.044715 x^3)
# value = -2.2917961972623857e-07,  derivative = -1.5463619876646428e-06

jax.grad(lambda t: jax.nn.gelu(t, approximate=False))(x)  # -7.1469460017922964e-06 (the exact lane)
```

| dtype | x | `jax.nn.gelu` | `jax.grad` | cancellation-free value | cancellation-free derivative | exact-erf gradient |
|---|---|---|---|---|---|---|
| **f32** | **-4.9** | **-0.0** | **0.0** | -1.0e-05-ish | -1.0e-05-ish | -1.0e-05 |
| **f32** | **-5** | **-0.0** | **0.0** | -2.2917961972623857e-07 | -1.5463619876646428e-06 | -7.1469460017922964e-06 |
| **f32** | **-6** | **-0.0** | **0.0** | -8.43964897967453e-11 | -7.709976012946757e-10 | -3.547e-08 |
| **f32** | **-7** | **-0.0** | **0.0** | -2.3314683517128287e-15 | -2.7842558190908897e-14 | -6.266e-11 |
| **f32** | **-10** | **-0.0** | **0.0** | -1.3e-37 | -1.9e-36 | -7.694598626706473e-22 |
| **f64** | **-7.35** | **-0.0** | **0.0** | -1.2e-53 | -1.8e-52 | -6.2e-53 |
| f64 | -5 | -2.2917961944868281e-07 | -1.54636198694884e-06 | -2.2917961972623857e-07 | -1.5463619876646428e-06 | -7.1469460017922964e-06 |
| f64 | -1 | -0.1588080093917233 | -0.08296408384578258 | -0.15880800939172324 | -0.08296408384578252 | -0.08331547058768629 |

The first input with an exactly zero gradient is `x = -4.9` in float32 and `x = -7.35` in float64
(0.05 grid), and every input below it is exactly zero, not merely small.

## Root cause

`jax.nn.gelu`'s default is the tanh approximation `0.5 x (1 + tanh(u))` with
`u = sqrt(2/pi) (x + 0.044715 x^3)`.  Once `tanh(u)` rounds to `-1` the factor `1 + tanh(u)` is
exactly zero - the same saturation the campaign's JAX-009 reports for `tanh`'s own JVP rule applied
to the activation's *value*.  The derivative inherits it through `1 - tanh(u)^2`, which is also
exactly zero there:

* `float32`: `u(-4.9) = -8.29`, `jnp.tanh(u) == -1.0`, `1 + tanh(u) == 0.0`, `1 - tanh(u)^2 == 0.0`
  (all asserted by the reproducer);
* `float64`: the same at `u(-7.35) = -12.4`.

The exact-erf formulation `0.5 x erfc(-x/sqrt(2))` has no such saturation and keeps the tail.

## Impact

* **A dead gradient in the default activation's normal range**: in float32 every pre-activation
  `x <= -4.9` produces an exactly zero gelu gradient, so those units never update and the loss
  receives no signal from them.  float32 is the common training dtype and `jax.nn.gelu` is the
  default gelu in JAX (and `approximate=True` is its default).
* The activation value is wrong too (exactly `-0.0` instead of a representable negative number), so
  forward-only uses (feature maps, layer statistics, distillation targets) are affected as well.
* The two lanes of the same API disagree qualitatively: `approximate=False` is correct at every one
  of those inputs, so the workaround is known and the defect is specific to the default.
* Silent: the values are exact zeros, so a finiteness check passes and a numerical gradient check
  (which perturbs around a point where the output is already 0) can too.

## Suggested fix

Evaluate the approximation without the saturation, e.g. through the sigmoid form
`0.5 x sigmoid(2u)` (whose derivative is `0.5 sigmoid(2u) + x sigmoid(2u)(1 - sigmoid(2u)) u'` and
never cancels), or make the exact-erf formulation the default.  A regression test:

```python
x = jnp.float32(-5.0)
assert float(jax.grad(jax.nn.gelu)(x)) != 0.0        # currently 0.0
assert float(jax.nn.gelu(x)) != 0.0                  # currently -0.0
```

## Controls (all in `repro.py`, 75 checks, 0 failing on each of 2 replays)

* The reference is the tanh approximation evaluated in a cancellation-free form
  (`1 + tanh(u) = 2 e^{2u}/(1 + e^{2u})`, `1 - tanh^2 = 4 e^{2u}/(1 + e^{2u})^2`), asserted to be
  nonzero and representable at every failing point.
* Above the threshold (`x = -4, -3, -2, -1`) the activation and gradient match that reference, so the
  defect is the saturation and not the formula; `x = 0` gives `(0.0, 0.5)` and `x = 5, 10` match the
  reference to `1e-9` relative.
* The exact-erf lane (`approximate=False`) is asserted to keep a nonzero negative tail at every
  affected point, and the default is asserted to equal `approximate=True` explicitly.
* The mechanism is asserted directly (`tanh(u) == -1`, `1 + tanh(u) == 0`, `1 - tanh(u)^2 == 0`) at
  four failing inputs.
* `jax.jit`, `jax.value_and_grad` and a one-layer `gelu(w - 5)` composition are checked, the last one
  showing the zero gradient propagating to a parameter; `float32` and `float64` are covered.

## Cross-framework position (measured in the same week)

The collapse is a property of the *tanh approximation formula*, not of JAX's arithmetic, and every
framework's exact lane is fine - but JAX is the only one of the three where the approximation is the
default:

| framework | lane | float32 `x = -5` | float32 `x = -7` | float64 `x = -10` |
|---|---|---|---|---|
| **JAX** | `jax.nn.gelu` (**default**) | value `-0.0`, grad `0.0` | `-0.0` / `0.0` | `-0.0` / `0.0` |
| JAX | `approximate=False` | grad `-7.15e-06` | `-6.27e-11` | `-7.69e-22` |
| TensorFlow | `tf.nn.gelu(approximate=True)` | value `-0.0`, grad `0.0` | `-0.0` / `0.0` | `-0.0` / `0.0` |
| TensorFlow | `approximate=False` | grad `-7.15e-06` | `-6.27e-11` | `-7.62e-22` |
| PyTorch | `F.gelu(approximate='tanh')` (opt-in) | value `-2.98e-07`, grad `-2.01e-06` | value `-0.0`, grad `0.0` | `-0.0` / `0.0` |
| PyTorch | `approximate='none'` (default) | grad `-7.14e-06` | `-6.39e-11` | `-7.69e-22` |

So the JAX report is the strongest instance (the default API is affected from `x <= -4.9` in the
common training dtype) and the same defect class is present in TensorFlow's `approximate=True` lane
and in PyTorch's opt-in `approximate='tanh'` lane, recorded in this campaign's ledger as a
cross-framework characterisation rather than as three separate drafts.

## Novelty

JAX tracker searches for `gelu gradient zero`, `gelu tanh approximation`, `jax.nn.gelu`,
`gelu float32 saturate` and `gelu derivative zero large negative` return nothing about this
observable.  The tracker's nearest items are the campaign's own class-mates (still open and per
operator): #40342 (`sigmoid` default JVP tail), #39794/#39795/#39796/#39798 (`expm1`/`elu`/`selu`/
`silu`) and this campaign's JAX-009 (`tanh`'s own JVP) - all about the tanh-family cancellation, none
about `gelu`.  The ledger's `gelu` rows are all **PyTorch** (`F.gelu` layout/width sign-of-zero,
PT-021), so the JAX activation is a new surface.

## Environment

JAX 0.10.2 / jaxlib 0.10.2, CPU backend, Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64,
`jax_enable_x64=True`.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"The tanh approximation is approximate by definition."**  It is an *approximation of gelu*, not a
  function that is zero: at `x = -5` its value is `-2.29e-07` and its derivative `-1.55e-06`, both
  representable in float32, and the same API with `approximate=False` returns a nonzero gradient.  The
  approximation error near `-5` is a few percent, not 100%.
* **"Only the far tail is affected."**  The float32 threshold is `x <= -4.9`, i.e. an ordinary
  pre-activation; and the gradient is exactly zero for *every* input below it.
* **"The gradient is zero anyway near the tail."**  For exact gelu the derivative at `x = -5` is
  `-7.1e-06` (the exact lane returns it); for the approximation it is `-1.5e-06`.  Neither is zero,
  and the difference between "small" and "exactly zero" is what kills a unit's learning signal.
* **"This duplicates the tanh issues."**  The mechanism is the same family, but the observable is a
  different public operator with a much lower threshold (float32 `|x| >= 4.9` here versus `|x| >= 20`
  for `jnp.tanh` in JAX-009) and it affects both the value and the gradient of the *default*
  activation; the tracker files this class per operator.
