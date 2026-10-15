# `tf.math.maximum` / `tf.math.minimum` disagree with each other and with IEEE 754 on the sign of zero, and the sign flips with tensor width

**Component:** TensorFlow 2.21.0, CPU, eager — `tf.math.maximum`, `tf.math.minimum`, and everything
built on them (`tf.nn.relu6`, `tf.minimum(tf.maximum(x, 0.), 6.)`, clipping idioms).
**Severity:** silent wrong sign of zero, which is *not* cosmetic: it propagates to `±inf` through
`1/x` (and through `log`, `atan2(y, ...)`, `copysign`, ...).  The same expression also returns a
different sign depending on how many elements the tensor has.
**Found with:** Rondo's vectorisation-width oracle (`artifact/hunt-v2/tools_tf_special_width.py`,
class-only comparison of the Eigen scalar functor against the SIMD packet functor).

---

## Summary

For the two signed zeros, IEEE 754-2019 specifies the result of `maximum`/`minimum`:

```text
maximum(+0, -0) = +0        minimum(+0, -0) = -0
maximum(-0, +0) = +0        minimum(-0, +0) = -0
```

TensorFlow's CPU kernels return different signs for a 1-element tensor and for a tensor with one SIMD
packet or more, and neither width implements the specification for both operand orders:

```python
import numpy as np, tensorflow as tf
for w in (1, 4, 8, 64):
    x = tf.constant([0.0] * w); y = tf.constant([-0.0] * w)
    m = tf.math.maximum(x, y).numpy()[0]
    print(f"n={w:<3} maximum(+0,-0) = {m!r:>5} signbit={bool(np.signbit(m))}")
```

```text
n=1   maximum(+0,-0) =  0.0 signbit=False      <- matches IEEE
n=4   maximum(+0,-0) =  0.0 signbit=False      <- matches IEEE
n=8   maximum(+0,-0) = -0.0 signbit=True       <- violates IEEE (the SIMD packet functor)
n=64  maximum(+0,-0) = -0.0 signbit=True
```

The threshold is exactly one SIMD packet, and the two functors use *opposite tie-breaks*: the scalar
functor returns its first operand on a tie, the packet functor returns its second.  Reversing the
operands therefore moves the violation to the other width:

| expression | n=1 | n=4 | n=8 | n=64 | IEEE 754-2019 |
|---|---|---|---|---|---|
| `maximum(+0, -0)` | `+0.0` | `+0.0` | **`-0.0`** | **`-0.0`** | `+0.0` |
| `maximum(-0, +0)` | **`-0.0`** | **`-0.0`** | `+0.0` | `+0.0` | `+0.0` |
| `minimum(+0, -0)` | **`+0.0`** | **`+0.0`** | `-0.0` | `-0.0` | `-0.0` |
| `minimum(-0, +0)` | `-0.0` | `-0.0` | **`+0.0`** | **`+0.0`** | `-0.0` |

(bold = contradicts IEEE 754-2019 for that operand order)

### The same defect exists in the float64 lane, with the packet boundary at 4

Measured independently (the fp64 SIMD packet holds half as many lanes):

| expression | n=1 | n=2 | n=4 | n=8 | n>=64 | IEEE 754-2019 |
|---|---|---|---|---|---|---|
| `maximum(+0, -0)` | `+0.0` | `+0.0` | **`-0.0`** | **`-0.0`** | **`-0.0`** | `+0.0` |
| `maximum(-0, +0)` | **`-0.0`** | **`-0.0`** | `+0.0` | `+0.0` | `+0.0` | `+0.0` |
| `minimum(+0, -0)` | **`+0.0`** | **`+0.0`** | `-0.0` | `-0.0` | `-0.0` | `-0.0` |
| `minimum(-0, +0)` | `-0.0` | `-0.0` | **`+0.0`** | **`+0.0`** | **`+0.0`** | `-0.0` |

`tf.nn.relu6(-0.0)` likewise gives `-0.0` up to `n=2` and `+0.0` from `n=4` in fp64, so
`1/relu6(-0.0)` is `-inf` vs `+inf` - the same consequence in both lanes.  A fix therefore has to
cover the fp64 functor pair as well; the reproducer asserts both dtypes.

## Why it matters

The sign of zero decides the sign of infinities downstream, so this is an observable value change,
not a formatting detail.  A standard activation composite inherits it:

```python
relu6 = tf.nn.relu6(tf.constant([-0.0] * 8))[0]      # +0.0   (packet path, IEEE-correct here)
inv   = 1.0 / relu6                                   # +inf
relu6_1 = tf.nn.relu6(tf.constant([-0.0]))[0]         # -0.0   (scalar path)
inv_1   = 1.0 / relu6_1                               # -inf
```

So `1.0 / tf.nn.relu6(x)` changes sign at the packet boundary for `x = -0.0`, and a model that is
correct with 8 elements is wrong with 4 (or the other way round).  `tf.math.maximum`/`minimum` are
also what `tf.clip_by_value`, `tf.nn.relu`, `tf.nn.relu6`, leaky-ReLU style code and many custom
losses are written with, so the discrepancy is reachable from ordinary user code.

`tf.nn.relu6` shows the same flip (the fused kernel and the explicit
`tf.minimum(tf.maximum(x, 0.), 6.)` composite behave identically), which confirms it is `maximum`
inside the kernel rather than something relu6-specific.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).` (each replay runs in its own process)

The reproducer asserts:

* the sign of `maximum`/`minimum` over the signed zeros differs between width 1 and width 8 for every
  operand order (4/4 cases), with the flip at exactly one packet;
* for each operand order, the width whose answer contradicts IEEE 754-2019 is identified, and the
  reproducer checks that *some* width is wrong for every order (the two functors cannot both be right);
* `tf.nn.relu6(-0.0)` inherits the flip, and `1.0 / tf.nn.relu6(-0.0)` is `-inf` on the scalar path
  and `+inf` on the packet path;
* controls that must not move: `maximum(1, -1)=1`, `minimum(1, -1)=-1`,
  `relu6(nan)=nan`, `relu6(inf)=6`, and the packed/clamped values `relu6(7)=6`, `relu6(1)=1`.

## Root cause

Eigen's `scalar_max_op`/`scalar_min_op` and their packet counterparts in `cwise_ops` are separate
implementations.  The scalar functor uses `numext::maxi(a, b)`-style logic that returns the first
argument on a tie; the packet functor (via `pmax`/`pmin` against the packet constant) returns the
second.  Neither applies the signed-zero rule of IEEE 754-2019, where the choice of zero is
specified rather than left to the tie-break.  This is the same mechanism as TF-001 in this
repository (`tf.math.xlogy(-0.0, y)`: `+0.0` below one packet, `-0.0` at/above it): one op, two
implementations, different answers.

## Novelty assessment

* Searches on `tensorflow/tensorflow` for `maximum negative zero`, `minimum negative zero`,
  `sign of zero maximum`, `relu6 signed zero`, `relu6 negative zero`, `maximum 0.0 -0.0 tie` and
  `signed zero maximum packet functor` return no matching report.  The known XLA NaN family
  (#116992, #118077, #118643) is about NaN, not about the sign of zero, and is eager-vs-XLA rather
  than eager-scalar-vs-eager-packet.
* The defect class is recognised as a real bug by other projects, which independently fixed exactly
  this: `AmbiqAI/ns-cmsis-nn#316` ("arm_minmax_f32/f16: scalar and MVE legs return opposite signed
  zeros") and `tile-ai/TileOPs#469` ("Restore signed-zero-correct maximum/minimum semantics").
  This is a different framework, so it does not cover TensorFlow, but it shows the issue is
  considered worth fixing rather than a cosmetic difference.
* The same defect class exists in PyTorch's CPU path with *its* threshold: `torch.maximum` returns
  the first operand on a tie up to `n=8` and the second from `n=16` onwards (measured; NumPy is
  width-stable but also returns the second operand, so it too contradicts the IEEE result for one
  operand order at every width).  That makes this a property of the scalar-vs-vector tie-break
  rather than of TensorFlow alone - but it does not cover TensorFlow, whose own two functors
  disagree at TensorFlow's own packet width, and the PyTorch report would be a separate patch in a
  separate codebase.  Being aware of it also means the fix here should not simply copy whichever
  tie-break PyTorch/NumPy uses.
* `mindspore-ai/mindspore#381` reports `ops.relu6` replacing NaN with 0.0 - a different symptom in a
  different framework, and not about the sign of zero.

## Files

* `repro.py` - self-checking reproducer (4 operand orders x width scan, the IEEE verdict per width,
  the relu6 + reciprocal consequence, the controls, two process-isolated replays)
* `evidence.txt` - raw observations
* `collect_env.txt` - environment

## Additional triggers (wave 131)

The same scalar functor feeds `tf.nn.relu` and `tf.nn.relu6`, and `tf.math.xdivy` flips the sign the
other way.  All three flip at exactly the packet boundary this report is about — `n = 8` for float32
and `n = 4` for float64 — and the `relu`/`relu6`/`xdivy` checks are asserted by `repro.py` now
(192 checks, 2/2 replays):

| op | input | scalar widths (n < packet) | packet widths (n >= packet) | IEEE-754-2019 |
|---|---|---|---|---|
| `tf.nn.relu` | `-0.0` | `-0.0` | `+0.0` | `+0.0` (as `maximum(x, 0)`) |
| `tf.nn.relu6` | `-0.0` | `-0.0` | `+0.0` | `+0.0` |
| `tf.math.xdivy` | `(-0.0, 1.0)` | `+0.0` | `-0.0` | `-0.0` |
| `tf.math.maximum` | `(-0.0, 0.0)` | `-0.0` | `+0.0` | `+0.0` |

In every row the **scalar path is the one that violates IEEE-754**, which is the same conclusion the
`maximum`/`minimum` cases reach above — the defect is in the scalar functor, not in the vectorised one,
and widening a tensor from 7 to 8 elements (float32) or 3 to 4 (float64) changes the sign of the
result.  `tf.math.sign(-0.0)` stays `+0.0` in eager at every width; its width dependence is on the XLA
side and is already covered by `TF-004`.

Found by `artifact/hunt-v2/tools_tf_sign_packet_sweep.py`, which sweeps 27 ops x 4 dtypes x lengths
1..16 for sign-of-zero and class differences against an fp64 reference.

## The boundary is the total element count, not the contiguous run (wave 134)

A reshaped tensor holds the same values, so a natural question is whether the flip follows the *length
of a contiguous run* (the innermost dimension) rather than the total size - if it did, `(2, 4)` and
`(8,)` would disagree for float32.  They do not: `tools_tf_shape_packet_probe.py` checks 11 ops x 2
dtypes x 6 shapes of the same total size - `(n,)`, `(1, n)`, `(n, 1)`, `(2, n/2)`, `(n/2, 2)` and
`(2, 2, n/4)` - and finds **no shape dependence at all** (0/22 op/dtype combinations).  The sign is a
function of the total element count alone, so reshaping cannot avoid the defect and does not introduce
it either.  `repro.py` now asserts that invariance for `maximum` as well (200 checks, 2/2 replays), so a
future fix that only corrects the flat case will fail the suite.
