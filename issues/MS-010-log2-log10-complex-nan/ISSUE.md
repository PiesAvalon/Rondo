# `ops.log2` / `ops.log10` return a `nan` imaginary part for complex inputs with a zero or infinite real part, while `ops.log` is correct

**Component:** MindSpore 2.9.0, CPU, PYNATIVE — `ops.log2` and `ops.log10` on `complex64` / `complex128`
(the compiled `ms.jit` path behaves identically, so this is not a mode-specific defect)
**Severity:** silent `nan` where the correct value is a finite or zero imaginary part; the NaN
propagates to anything that touches the imaginary component
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** the complex branch-cut sweep
(`artifact/hunt-v2/tools_ms_complex_worker.py`, eager and `ms.jit` against a NumPy C99 reference);
certified reproducer `repro.py` in this directory,
`python repro.py --replays 2` -> `RESULT: 0 failing check(s).`

---

## Summary

For complex inputs whose real part is zero or infinite, the imaginary part of the binary logarithm is
`0` (or `±pi/ln 2` for `-0`), and every other implementation returns it: NumPy (the C99 reference),
PyTorch and TensorFlow.  MindSpore's `ops.log2` and `ops.log10` return `nan` instead, while
`ops.log` — the same family and the function the other two are mathematically built from — is
correct.  `complex64` and `complex128` behave identically.

```python
import numpy as np, mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
pts = [complex(0.0, 0.0), complex(0.0, -0.0), complex(-0.0, 0.0),
       complex(float("inf"), 0.0), complex(float("-inf"), 0.0)]
x = ms.Tensor(np.array(pts, np.complex64))
print(np.asarray(ms.ops.log2(x).asnumpy()))
print(np.log2(np.array(pts, np.complex64)))
```

| input | `ops.log2` (MindSpore) | `np.log2` (reference) |
|---|---|---|
| `+0+0j` | `-inf+nanj` | `-inf+0j` |
| `+0-0j` | `-inf+nanj` | `-inf-0j` |
| `-0+0j` | `-inf+nanj` | `-inf+4.532360076904297j` |
| `-0-0j` | `-inf+nanj` | `-inf-4.532360076904297j` |
| `+inf+0j` | `inf+nanj` | `inf+0j` |
| `-inf+0j` | `inf+nanj` | `inf+4.532360076904297j` |

`ops.log10` shows the same pattern with `pi/ln 10 = 1.364376425743103`, and `ops.log` matches NumPy
at all six points — the reproducer asserts all of this, plus the real parts, which stay correct.

## Why this is a defect and not a convention

* **MindSpore disagrees with itself.**  `ops.log` returns `-inf+0j`, `-inf±4.532j` and `inf+0j` for
  the same inputs; only the `log2`/`log10` wrappers lose the imaginary part.  If losing it were the
  intended convention, `log` would lose it too.
* **NumPy, PyTorch and TensorFlow all return the imaginary part**, and C99 Annex G defines
  `clog(+0 + 0i) = -inf + 0i` (with a divide-by-zero exception) — the zero is specified, not
  incidental.
* **A NaN is not a rounding difference.**  It propagates: summing the result over the same array
  gives a finite real part but a NaN imaginary part in MindSpore, where NumPy stays finite, and any
  later `abs`, comparison or gradient that reads the imaginary component becomes NaN.

## Scope

Only inputs with `real == 0` or `|real| == inf` are affected.  Ordinary negative reals such as
`log2(-1+0j) = 4.532360076904297j` and `log2(-4+0j) = 2 + 4.532360076904297j` are **correct** in
MindSpore, as are positive inputs; the sweep that found this covered 21 complex ops on a 14-point
branch-cut grid plus a length dimension, and the only substantive deviations were these two
functions (the remaining differences were 1-ULP real parts and signs of zeros).

## Reproduction

```bash
python repro.py --replays 2      # RESULT: 0 failing check(s).
```

28 checks covering: the `log` control, the three points where NumPy's imaginary part is exactly zero,
all six zero/infinite-real inputs, the real parts, the NumPy reference having no NaN, and NaN
propagation through a sum.  Both replays run in their own process.

## Novelty assessment

Searches on the MindSpore issue mirror (`mindspore-ai/mindspore`) for `log2 complex nan`,
`log10 complex`, `complex log2 imaginary` and `complex log nan` return no matching report, and the
campaign's earlier MindSpore searches (`denormal`, `subnormal`, `FTZ`, `非规格化`, `次正规`,
`complex`, `sqrt`) found nothing covering complex logarithm edge cases either.  The nine MindSpore
drafts already in this package are a different family (float64 inputs computed with float32
intermediates, state-dependent subnormal flushing, and one GRAPH-mode `fmod` defect); none is about
the imaginary part of a complex logarithm.

## Proposed fix sketch

Implement `log2`/`log10` as `ops.log(z)` divided by the real constant `ln 2` / `ln 10` (using the
same complex `log` that is already correct), instead of whatever wrapper currently drops the
imaginary component; add a regression test over `{0±0j, -0±0j, ±inf+0j}` for both complex dtypes.
