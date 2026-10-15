# The `ops` logaddexp family (`logaddexp`, `logaddexp2`, `logsumexp`) returns `nan` for infinite inputs, while `mnp` returns `inf`/`-inf`

**Component:** MindSpore 2.9.0, CPU, PYNATIVE and GRAPH (`ms.jit`), `ops.logaddexp`,
`ops.logaddexp2` and `ops.logsumexp` on `float32`/`float64`
**Severity:** silent wrong value — `nan` where the mathematical (and NumPy/PyTorch/JAX, and
MindSpore's own `mnp`) result is `+inf` or `-inf`; the `nan` then poisons every downstream reduction
**Status:** present in MindSpore 2.9.0 (CPU); `mnp.logaddexp`/`mnp.logaddexp2` in the same build are
correct for the same inputs
**Found with:** the wave-165 stepping/scaling grid (`artifact/hunt-v2/results/w165_*`) and extended
by the wave-168 reduction grid (`artifact/hunt-v2/results/w168_*`), triaged by Rondo's
`check_consensus` oracle with NumPy as arbiter; the certified reproducer is `repro.py` in this
directory (72 self-checking assertions, `--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

`ops.logaddexp` and `ops.logaddexp2` return `nan` when **both** operands are the same infinity:

```python
import numpy as np, mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
def T(x): return ms.Tensor(np.asarray(x), ms.float64)

print(ops.logaddexp(T(np.inf), T(np.inf)).asnumpy())     # nan   <- log(exp(inf)+exp(inf)) = inf
print(ops.logaddexp(T(-np.inf), T(-np.inf)).asnumpy())   # nan   <- -inf
print(ops.logaddexp2(T(np.inf), T(np.inf)).asnumpy())    # nan   <- inf
print(ops.logaddexp2(T(-np.inf), T(-np.inf)).asnumpy())  # nan   <- -inf
print(ops.logsumexp(T(np.asarray([np.inf, 0.0])), 0).asnumpy())      # nan   <- inf
print(ops.logsumexp(T(np.asarray([-np.inf, -np.inf])), 0).asnumpy()) # nan   <- -inf
```

```text
                          ops (2.9.0)   mnp (2.9.0)   numpy / torch / jax
logaddexp( +inf,  +inf)      nan            +inf            +inf
logaddexp(-inf,  -inf)       nan            -inf            -inf
logaddexp2(+inf,  +inf)      nan            +inf            +inf
logaddexp2(-inf,  -inf)      nan            -inf            -inf
logsumexp([-inf])            nan             -inf            -inf
logsumexp([ inf, 0])         nan             +inf            +inf
logsumexp([-inf, -inf])      nan             -inf            -inf
logsumexp([ inf, -inf])      nan             +inf            +inf
```

`ops.logsumexp` - the reduction built on the same kernel - is `nan` for **every** array that
contains an infinity or consists only of `-inf` (8 of the 23 arrays in the wave-168 grid, including
`[inf]`, `[inf, 0]`, `[inf, inf]`, `[inf, -inf]`, `[-inf]`, `[-inf, -inf]` and `[inf, 1, -inf, 2]`),
while its finite inputs reduce correctly and `numpy.logaddexp.reduce` returns the right limits.

Every other special-value combination is correct in `ops`: `(inf, -inf)`, `(inf, 0)`, `(-inf, 0)`,
`(inf, 1)`, `(nan, inf)`, `(nan, 0)`, `(0, 0)`, the finite large pairs, and the whole ordinary range.
The graph kernel (`ms.jit`) has the defect too, the vector form fails elementwise, and both
widths are affected.  `mnp.logaddexp`/`mnp.logaddexp2` return `inf`/`-inf` for exactly the same
inputs, so MindSpore's two public APIs disagree with each other.

## Root cause

The `ops`/`mint` kernel evidently takes the path `max(a, b) + log1p(exp(-|a-b|))` without special
casing equal infinities: for `(inf, inf)` the difference `a - b` is `inf - inf = nan`, and the `nan`
propagates out of the `log1p` even though the leading term is `inf`.  The `mnp` layer and the other
frameworks special-case it (the mathematically correct limit is `+inf` for equal `+inf` and `-inf`
for equal `-inf`).  `mint.logaddexp`/`mint.logaddexp2` currently raise `RuntimeError: The kernel
LogAddExp unregistered` (the separate MS-013 mint-registration family), so only the `ops` layer can
be exercised here.

## Suggested fix

Short-circuit the special cases before the stable formula: if either operand is `+inf` the result is
`+inf`; if both are `-inf` the result is `-inf`; only then use
`max(a, b) + log1p(exp(-|a-b|))`.  A regression test should cover `(±inf, ±inf)` for both functions
and both widths - the `mnp` implementation already contains the correct behaviour.

## Impact

Log-space arithmetic (mixture models, HMMs, log-probability accumulation) uses `logaddexp`
precisely on values that may saturate to `-inf` (impossible events) or `+inf` (overflowing scores).
Combining two saturated values should keep the saturation; instead both become `nan`, which then
propagates through the whole accumulation and is far harder to localise than an `inf`.

## Controls (all in `repro.py`)

* `(inf, -inf)`, `(-inf, inf)`, `(inf, 0)`, `(-inf, 0)`, `(inf, 1)`, `(-inf, 1)`, `(nan, inf)`,
  `(inf, nan)`, `(nan, 0)`, `(0, 0)`, `(1, 1)`, `(-1e308, -1e308)`, `(1000, 1000)`,
  `(-1000, -1000)` all match NumPy for both functions.
* `ops.logsumexp` on finite arrays (`[0]`, `[0, 0]`, `[1, 2]`, `[1000, 1000]`, `[-1000, -1000]`)
  matches `numpy.logaddexp.reduce`.
* `mnp.logaddexp`/`mnp.logaddexp2` return `inf`/`-inf` for the failing inputs (float32), proving the
  defect is in the `ops` kernel rather than in an unspecified convention.
* The finite element of a mixed vector (`[inf, -inf, 0]` combined with itself) is still correct, so
  the kernel is not globally broken.

## Novelty

Searched the MindSpore tracker for `logaddexp`, `logaddexp2`, `logaddexp inf`: the only hits are
mindspore#387 and mindspore#388, which report that **`mnp.logaddexp`/`mnp.logaddexp2` overflow to
`inf` for large finite float32 inputs** (a numerical-stability issue in the `mnp` layer, with float32
finite inputs).  This report is different in API layer (`ops`, not `mnp`), input class (equal
infinities, not large finite values) and symptom (`nan`, not `inf`); notably, `mnp` - the layer those
issues concern - is *correct* for our inputs, so this is an inconsistency between the two kernels
rather than a duplicate.  The campaign's other MindSpore `nan`-for-`inf` findings are other operators
(MS-010 complex `log2`/`log10`, MS-012 `polygamma(n, inf)`, MS-019 `fmod`/`remainder` with an
infinite divisor).

## Environment

MindSpore 2.9.0 CPU wheel, NumPy 1.26.4, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; `ops` vs `mnp` table in `evidence.txt`.

---

## Anticipated objections and responses

* **"`logaddexp(inf, inf)` is a degenerate input."**  It is the natural result of saturated
  log-probabilities, and the answer is not ambiguous: NumPy, PyTorch, JAX, MindSpore's own `mnp` and
  the mathematical limit all give `inf`.  A `nan` there destroys a whole accumulation.
* **"This overlaps mindspore#387/#388."**  Those are `mnp.*` float32 *large finite* overflow
  reports; this is the `ops.*` kernel returning `nan` for *infinite* inputs, and `mnp.*` handles our
  inputs correctly, so the two cannot be fixed by the same change.
* **"`mint` is the future API."**  `mint.logaddexp`/`mint.logaddexp2` cannot even run in 2.9.0 (the
  MS-013 unregistered-kernel family), so `ops` is the only usable layer today and it is wrong.
