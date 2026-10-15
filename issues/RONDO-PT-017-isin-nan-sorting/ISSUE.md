# `[inductor]` `torch.isin` returns **False** for an `inf` needle when the test set contains `nan` (the sort + `searchsorted` decomposition path)

**Component:** `torch.compile` → Inductor, via the `aten.isin` decomposition in
`torch/_decomp/decompositions.py` (`isin` → `isin_sorting`)
**Severity:** **silent wrong result** (a membership test flips to False)
**Status:** present in `torch 2.12.1+cu130`; `aot_eager` / `eager` backends and CUDA are correct
**Found with:** Rondo edge-value OpInfo sweep (`opinfo_campaign.py --fill-edges`) + delta-debugging

---

## Summary

`torch/_decomp/decompositions.py::isin` selects between two algorithms:

```python
if test_elements.numel() < 10.0 * pow(elements.numel(), 0.145):
    return isin_default(elements, test_elements, invert=invert)   # broadcast equality
else:
    return isin_sorting(elements, test_elements, ...)             # sort + searchsorted
```

`isin_sorting` sorts the test elements and binary-searches the needles:

```python
sorted_test_elements, _ = torch.sort(test_elements_flat)
idx = torch.searchsorted(sorted_test_elements, elements_flat)
cmp = sorted_test_elements[torch.where(idx < n, idx, 0)] == elements_flat
```

`torch.sort` places `nan` last, so the sorted array ends with `[..., inf, nan]`.
`torch.searchsorted` is a binary search; because every comparison against `nan` is false, it
treats the trailing `nan` as "not less than" the `+inf` needle and returns the insertion point
`n` (past the end).  The decomposition then clamps that with
`torch.where(idx < n, idx, 0)` to index `0`, whose element is the smallest value (`-inf` here),
so `sorted_test_elements[idx] == inf` is False.  Inductor applies this decomposition, so a
compiled membership test answers False where eager answers True.

```python
import torch
inf, nan = float("inf"), float("nan")
a = torch.tensor([inf])
b = torch.tensor([0.0, 1.0, -1.0, inf, -inf, nan, 1e-38, 3.4e38, 0.5, -0.5, 0.0, 1.0])

torch.isin(a, b)                                   # tensor([True])
torch.compile(lambda x, y: torch.isin(x, y))(a, b)  # tensor([False])   <-- wrong
```

## Reproduction results

`python repro.py --replays 2 --scan` (in this directory): 28 checks, 0 failures, identical in two
runs.

Triggers:

| case | eager | compiled |
|---|---|---|
| `isin([inf], 12-element set with inf+nan)` | `[True]` | `[False]` |
| same in float64 | `[True]` | `[False]` |
| 13-element set | `[True]` | `[False]` |
| 20-element edge cycle | `[True]` | `[False]` |
| `isin(5 elements, 20-element cycle)` (the OpInfo-shaped case) | `[True, True, True, True, True]` | `[True, True, True, **False**, True]` |

Controls (all correct — they isolate the sorting path and the `+inf` needle):

| control | result |
|---|---|
| 10 `elements` (threshold `10·10^0.145 ≈ 13.96 > 12`) → broadcast path | match |
| `assume_unique=True` (different branch of `isin_sorting`) | match |
| test set without `nan` | match |
| needle `-inf`, `3.4e38`, `1e-38`, `1.0` | match |
| `backend="aot_eager"`, `backend="eager"` | match |
| CUDA | match |

The length scan in `evidence.txt` (`n = 4..20`) shows the failure depends on the test-set length:
`n = 11, 12, 13, 19, 20` answer incorrectly while `n = 4..10` and `n = 14..18` are correct.
The result is length-sensitive because whether `nan` and the `inf` needle fall into the same
binary-search interval depends on the layout after the sort.

## Root cause

The `isin_sorting` path assumes the sorted test array is totally ordered w.r.t. the needles.
With `nan` present that assumption fails: `nan` sorts to the end, and `searchsorted`'s binary
search cannot decide where `inf` sits relative to it, so it returns an index whose element is not
`inf`, and the subsequent equality check produces False.  Eager's implementation
(`aten::isin`) does not use this sort/searchsorted construction, so it is unaffected, and
`assume_unique=True` takes a different branch (concatenate + stable sort + duplicate mask) which
is also unaffected.

## Suggested fix

Handle the non-comparable entries explicitly in `isin_sorting`, e.g.

* compute the membership with `(elements.view(...) == test_elements).any(...)` whenever either
  side contains `nan`/`inf` (or split the test set into finite/non-finite partitions and test the
  non-finite needles with `isnan`/`isinf`), or
* use `torch.searchsorted` only for the finite part and handle `±inf`/`nan` separately;
* `torch.unique(test_elements)` before sorting would also remove the duplicate-driven length
  sensitivity but does not by itself fix the `nan` binary-search problem.

A regression test is small: the 12-element reproducer above, in float32 and float64, plus a
length scan around the algorithm-selection threshold.

## Related upstream work (novelty assessment)

* Tracker searches for `isin inf nan torch.compile`, `torch.isin wrong result inductor`,
  `isin nan compiled`, `isin membership inf inductor` and `isin searchsorted` return no matching
  report, and no issue or PR mentions `torch.isin` together with `nan`.
* **PR #147519** (`[Inductor] Fix the decompositions of torch isin`, **closed, unmerged**) touched
  the same function but fixed two *different* problems: scalar `test_elements` support (#147329)
  and `elements = 1, test_elements = [1, 2, 3, 4]` disagreeing with eager.  Neither involves
  `nan`/`inf`, and the build here still reproduces this report.
* **Issue #158738** (`searchsorted handling of nan`, open) is about
  `torch.searchsorted`'s *own* return value when the sorted sequence contains `nan` (it asks for
  `4` to be inserted before the trailing `nan`, i.e. the same index-`n` behaviour shown here).
  That issue never mentions `torch.isin`, `torch.compile` or Inductor.  It is also not the root
  cause of this report: `searchsorted` is *identical* between eager and compiled here
  (`repro.py --scan` prints the control), and `isin` can be fixed without changing `searchsorted`
  semantics — the decomposition must treat an out-of-range index as "not found" instead of
  clamping it to `0` and comparing.  Whatever the eventual resolution of #158738, the
  decomposition remains wrong until it stops assuming `idx < n`.
* #184893 / #185157 (bool `argmin`/`argmax` indices) and #185369 (adaptive pooling symbolic sizes)
  are different ops reached by the same edge-value sweep.
* This is the first finding from the new edge-value OpInfo mode
  (`opinfo_campaign.py --fill-edges`), which fills OpInfo samples with poles, signed zeros,
  infinities, NaNs and subnormals instead of well-behaved random data.

## Impact

`torch.isin` is a membership primitive; returning False for a value that is present is a silent
wrong answer.  The trigger needs `nan` in the test set and an `inf` needle — both are common in
data cleaning / masking code that mixes NaN sentinels with infinity bounds — and the failure is
additionally length-dependent, so it will not show up in small unit tests.

## Files

* `repro.py` — 5 triggers + 7 controls + backend/CUDA controls + a `--scan` length scan
* `evidence.txt` — the length scan, the algorithm-selection pivot and the decomposition source
* `collect_env.txt` — environment
