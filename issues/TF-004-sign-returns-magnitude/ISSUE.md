# `tf.math.sign` returns `|x|` for negative subnormal inputs on the vectorised path - a value outside its own codomain

**Component:** TensorFlow 2.21.0, CPU, eager — `tf.math.sign` / `tf.sign`, `float32` and `float64`.
**Severity:** silent wrong value.  `sign` is documented to return an element from `{-1, 0, 1}`; the
vectorised eigth-element path returns the **magnitude of the input**, i.e. a tiny positive number even
for negative input.  `tf.sign(x) * 5.0` therefore gives `~5e-38` instead of `-5.0`.
**Found with:** Rondo's sign-sensitive value probe (`artifact/hunt-v2/tools_sign_sensitive_multi.py`),
which compares raw values and sign bits rather than classes.

---

## Summary

For a subnormal input, `tf.math.sign` gives three different answers depending on the path, and none of
them is the documented `-1`:

| input | scalar path | vectorised path | reference |
|---|---|---|---|
| `-1e-38` (fp32 subnormal, packet width 8) | `0.0` | **`9.999999350456404e-39`** (= `\|x\|`) | `-1.0` |
| `1e-38` | `0.0` | **`9.999999350456404e-39`** (= `\|x\|`) | `1.0` |
| `-1e-320` (fp64 subnormal, packet width 4) | `0.0` | **`1e-320`** (= `\|x\|`) | `-1.0` |
| `-1.2e-38` (smallest fp32 *normal*) | `-1.0` | `-1.0` | `-1.0` |
| `-1e-40` (bfloat16 subnormal) | `0.0` | `0.0` | `-1.0` |
| `-6e-8` (float16 subnormal) | `-1.0` | `-1.0` | `-1.0` |

The defect is therefore fp32- and fp64-only; bfloat16 returns `0.0` (consistent with the flush policy)
and float16 is correct.

The complete path matrix shows where the wrong value lives:

| input | eager scalar | eager packet | `tf.function` (no XLA) | XLA | NumPy |
|---|---|---|---|---|---|
| `sign(-1e-38)` fp32 | `0.0` | **`9.999999350456404e-39`** | **`9.999999350456404e-39`** | `-0.0` | `-1.0` |
| `sign(1e-38)` fp32 | `0.0` | **`9.999999350456404e-39`** | **`9.999999350456404e-39`** | `0.0` | `1.0` |
| `sign(-1e-320)` fp64 | `0.0` | **`1e-320`** | **`1e-320`** | `-0.0` | `-1.0` |

`tf.function(jit_compile=False)` reproduces eager exactly, so the wrong value comes from the Eigen
kernel that both eager and the non-XLA graph use; the XLA path flushes to a **signed** zero, which is
the closest of the four answers to correct.  Normal inputs (`-1.2e-38` and above, `1.0`, `0.0`, `-0.0`)
are correct on all four paths.

```python
import tensorflow as tf
x = tf.fill((16,), tf.constant(-1e-38, tf.float32))
print(float(tf.math.sign(x)[0]))            # 9.999999350456404e-39   <-- not in {-1, 0, 1}
print(float(tf.math.sign(tf.fill((1,), tf.constant(-1e-38, tf.float32)))[0]))   # 0.0
print(float(tf.math.sign(tf.fill((16,), tf.constant(-1.2e-38, tf.float32)))[0]))  # -1.0 (normal input is fine)
```

The boundary is exactly one SIMD packet (7 elements behave like the scalar path, 8 and above like the
vectorised one), it reproduces in fp32 and fp64, and it is stable across processes.

## Why the "TensorFlow flushes denormals" policy does not explain it

TensorFlow deliberately flushes subnormal inputs and results (`denormal_test.py`,
`core/platform/denormal.cc`), so `0.0` on the scalar path - and the whole bfloat16 row - are consistent
with that policy.  `|x|` is **not**: a flushed input would give `sign(0)`, whose result is `0`, and a
correctly implemented `sign` can never return `1e-38`.  The vectorised kernel returns something outside
the function's contract, which the denormal policy cannot cover.  Note also that the vectorised result
is the *magnitude* for both signs of input, i.e. the direction of a negative number is lost.

## Impact, stated carefully

Inside TensorFlow this is largely masked: the wrong magnitude is a subnormal, so any downstream kernel
flushes it back to `0` - measured, `tf.math.sign(x) * 5.0` is `0.0` and `tf.math.sign(x) > 0` is
`False` even for `x = -1e-38` on the vectorised path.  A direction test therefore does **not** flip.

What is wrong is the value at the **API boundary**: `tf.math.sign(x).numpy()[0]` is
`9.999999350456404e-39` for `x = -1e-38` where the documented result is `-1`, and the tensor is
outside `sign`'s codomain `{-1, 0, 1}`.  That matters to anyone who consumes the sign outside
TensorFlow (a Python decision, NumPy, a non-TF library, an export path), and it is a width-dependent
inconsistency between the two eager kernels and the XLA path (`-0.0`).

## Reproduction

`python repro.py --replays 2` -> `RESULT: 0 failing check(s).` (each replay runs its own subprocesses;
one process per (dtype, width) combination, which is the only reliable form for compiled-path work in
this repository)

The reproducer asserts:

* fp32 and fp64: the scalar path returns `0.0` and the vectorised path returns exactly `|x|` for both
  signs of subnormal input, so the two paths disagree and the vectorised result is outside `{-1,0,1}`;
* the width boundary is one packet: `n = 1..7` agree with the scalar path, `n = 8` and above with the
  vectorised one;
* controls that must be correct everywhere: small *normal* inputs (`-1.2e-38`, `-1e-30`, `-1.0`,
  `1.0`, `0.0`, `-0.0`) return `-1.0`/`1.0`/`0.0` on both paths;
* the downstream consequence: `tf.math.sign(x) * 5.0` is `~5e-38` on the vectorised path instead of
  `-5.0` for a negative subnormal input.

## Novelty assessment

* Searches on `tensorflow/tensorflow` for `sign subnormal`, `tf.math.sign subnormal`, `sign denormal`,
  `sign wrong subnormal` and `sign returns input` return no report for real floating-point inputs.
* `tensorflow#116945` ("`tf.math.sign` returns 0 for small nonzero complex numbers (eager mode)",
  open) is the closest report, but it is **complex**-scoped and its symptom is `0`, not `|x|`; the
  real-dtype magnitude result reported here is a different defect.
* `tensorflow#121498` (open) mentions a subnormal in its title but is about `digamma` returning NaN.
* This is also distinct from the repository's TF-001/TF-003: those are sign-of-zero differences with
  both paths staying inside their codomain, whereas this returns a *magnitude* instead of a sign.

## Files

* `repro.py` - self-checking reproducer (one subprocess per (dtype, width), value-level assertions,
  controls, the downstream `sign(x)*5.0` consequence, two process-isolated replays)
* `evidence.txt` - raw observations
* `collect_env.txt` - environment
