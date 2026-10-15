# `ops.binary_cross_entropy` doubles the loss for a single-element input

**Component:** MindSpore 2.9.0, CPU: `ops.binary_cross_entropy` and `nn.BCELoss`, in `PYNATIVE_MODE`,
`GRAPH_MODE` and `ms.jit`, `float16` and `float32`, any shape with exactly one element (`(1,)`,
`(1, 1)`, `(1, 1, 1)`).
**Severity:** silent wrong loss value.  With one element, the `mean` and `sum` reductions return
**exactly twice** the documented loss while `reduction='none'` and every input with two or more
elements are correct and the **gradient is correct**, so training still runs - the reported loss is
just double, and it *changes when the batch size changes*: the same datum reports `2x` the mean loss
at batch size 1 and the correct mean at batch size 2.  Loss-combination weights, early stopping,
loss thresholds, gradient-clipping-by-loss scales and per-sample logging are all silently wrong for
single-sample steps.
**Status:** present in MindSpore 2.9.0 on CPU.  NumPy and PyTorch return the correct value at every
size (`torch.nn.functional.binary_cross_entropy` at size 1 is `0.6931471824645996` for the witness,
identical to its size-2 mean).
**Found with:** the wave-223 loss-reduction oracle (one-element vs multi-element reductions of the
same loss family against the closed-form value, with the `none` reduction as the internal
reference); certified reproducer `repro.py` in this directory (`--replays 2` ->
`RESULT: 0 failing replay(s).`, 38 checks).

---

## Summary

```python
import mindspore as ms
from mindspore import ops, Tensor
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

x = Tensor([[0.5]], ms.float32)          # documented loss: -[y log x + (1-y) log(1-x)] = 0.69314718
y = Tensor([[1.0]], ms.float32)

ops.binary_cross_entropy(x, y, reduction='none')   # [0.6931472]        correct
ops.binary_cross_entropy(x, y)                     # 1.3862944          2x  (mean)
ops.binary_cross_entropy(x, y, reduction='sum')    # 1.3862944          2x  (sum)

x2 = Tensor([0.5, 0.5], ms.float32)
y2 = Tensor([1.0, 1.0], ms.float32)
ops.binary_cross_entropy(x2, y2)                   # 0.6931472          correct
ops.binary_cross_entropy(x2, y2, reduction='sum')  # 1.3862944          correct
```

Measured (`float32`; `reduction='none'` and the per-element values are always correct):

| n | `reduction='none'` | `mean` | exact mean | `sum` | exact sum |
|---|---|---|---|---|---|
| **1** | `[0.6931472]` | **1.3862944** | 0.6931472 | **1.3862944** | 0.6931472 |
| 2 | `[0.6931472, 0.6931472]` | 0.6931472 | 0.6931472 | 1.3862944 | 1.3862944 |
| 3 | three copies | 0.6931472 | 0.6931472 | 2.0794415 | 2.0794415 |
| 4 | four copies | 0.6931472 | 0.6931472 | 2.7725887 | 2.7725887 |
| 5 | five copies | 0.6931472 | 0.6931472 | 3.4657360 | 3.4657360 |

The doubling is uniform in the inputs and follows the `weight`: with `weight=1.0` the result is
`1.3862943649291992` and with `weight=2.0` it is `2.7725887298583984`, i.e. twice the documented
weighted loss.  The shapes `(1,)`, `(1, 1)` and `(1, 1, 1)` behave identically, `float16` shows the
same factor (`1.38671875`), and the value is stable across `PYNATIVE_MODE`, `GRAPH_MODE`, `ms.jit`
and `nn.BCELoss`.

## Root cause

The per-element loss and the reductions for `n >= 2` are exact, and `reduction='none'` is exact for
`n = 1`, so the defect is in the scalar/reduced branch of `ops.binary_cross_entropy` when the input
has one element: it returns twice the loss it just computed.  The backward pass does not share the
error - `grad(mean)` is `[-2.0]` for one element and `[-1.0, -1.0]` for two, which are the exact
derivatives of the correct mean in both cases - so this is a forward-reduction defect, not a formula
constant.  (A factor of two in the formula would also double the `none` values and the gradient,
which it does not.)

## Impact

* Every single-sample loss report is 2x too high: online/streaming training, reinforcement-learning
  steps, per-example debugging loops, unit tests with one sample, and any `batch_size=1` epoch.
* The reported mean loss is **discontinuous in the batch size**: the same datum gives `1.3862944` as
  a batch of one and `0.6931472` as a batch of two, so any code that averages per-batch losses,
  compares a batch-1 probe against a batch-2 probe, or calibrates a threshold on a small batch sees
  a factor-of-two artefact.
* Because the gradient is correct, a loss-weighted multi-term objective is silently mis-scaled (for
  example `bce + mse`: only the BCE term doubles when the batch has one element) and a
  numerical-gradient check passes.
* `reduction='none'` gives the right per-element values, which is the workaround: call with
  `reduction='none'` and reduce manually.

## Suggested fix

Correct the single-element reduction path so that `mean`/`sum` return the per-element loss for a
one-element input (it is already computed correctly by the same call with `reduction='none'`).  A
regression test needs no reference library:

```python
x = ms.Tensor([[0.5]], ms.float32); y = ms.Tensor([[1.0]], ms.float32)
assert float(ops.binary_cross_entropy(x, y).asnumpy().reshape(-1)[0]) == \
       float(ops.binary_cross_entropy(x, y, reduction='none').asnumpy().reshape(-1)[0])
```

## Controls (all in `repro.py`, 38 checks, 0 failing on each of 2 replays)

* `reduction='none'` is asserted to be exactly `-log(0.5)` for `(1,)`, `(1, 1)` and `(1, 1, 1)`, so
  the per-element loss is known to be correct and the defect is localised to the reduction.
* Sizes 2, 3, 4, 5 and 16 are asserted to reduce correctly (`mean` and `sum`), including the exact
  multi-element values, so the defect is specific to `n = 1` rather than to the reduction in general.
* The **gradient** is asserted to be the exact derivative at `n = 1` and `n = 2`, which shows that
  the backward pass is not affected and rules out a formula-level factor of two.
* Seven other losses (`mse_loss`, `l1_loss`, `smooth_l1_loss`, `huber_loss`, `kl_div`,
  `binary_cross_entropy_with_logits`, `cross_entropy`) are checked to reduce a single element
  correctly, so the defect is specific to `binary_cross_entropy`.
* `nn.BCELoss` (all three reductions), `ms.jit` and `GRAPH_MODE` are checked to show the same values;
  `float16` is checked to show the same factor; the weight is checked to scale the doubled value.
* The batch-size discontinuity is asserted directly (one element vs two elements of the same datum).

## Novelty

MindSpore's tracker is on Gitee; searching it for `binary_cross_entropy` / `BCELoss` returns:

* [I3QOK6](https://gitee.com/mindspore/mindspore/issues/I3QOK6) (closed 2021-06-26) - "the value of
  BCEloss is different from pytorch", an Ascend 1.2 report whose body is two screenshots and no
  numbers.  If that was the same factor it was fixed long ago; the present report is a *specific*
  statement that the current 2.9.0 returns exactly 2x for a **one-element** input while `n >= 2` and
  `reduction='none'` are exact - a per-size reduction defect, not a general "value differs";
* [I5ZMPH](https://gitee.com/mindspore/mindspore/issues/I5ZMPH) (closed) - BCEloss differs between
  NPU and GPU and produced NaN on GPU (a backend difference on 1.9);
* [I7XIV0](https://gitee.com/mindspore/mindspore/issues/I7XIV0) (closed) - documentation quoting
  style for `ops.binary_cross_entropy`; plus dtype/parameter-compatibility requests
  (IC2NR7/IC2NPZ/IC2NOR, I7PJVU, I7O71Q).

None of them reports the single-element reduction, the `none`-vs-`mean` split, the correct gradient,
or the batch-size discontinuity.  The campaign's own ledger has two `binary_cross_entropy` rows: a
precision fuzz where it was clean (wave 112, multi-element) and an op-grid harness artifact that does
not concern this op (wave 194).

## Environment

MindSpore 2.9.0, CPU, Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64, `PYNATIVE_MODE` and
`GRAPH_MODE`.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"The difference is a factor of two in the BCE definition."**  A definition-level constant would
  also double the `reduction='none'` values, the `n >= 2` means/sums and the gradient; all four are
  correct (`none` is exactly `-log(0.5)`, the two-element mean is `0.6931472`, `grad(mean)` is
  `[-2.0]` for one element and `[-1.0, -1.0]` for two).  Only the reduced single-element path
  doubles.
* **"Only `batch_size=1` is affected."**  That is exactly the case the report is about, and it is a
  common one (online/streaming training, RL, probes, unit tests), the error is 100%, and it makes the
  loss depend on the batch size.
* **"The gradient is what matters, and it is correct."**  The gradient being correct is why the bug
  is silent; the loss value feeds early stopping, thresholding, multi-term weighting, logging and
  reported metrics, and a 2x scaling there is wrong.
* **"Already reported as I3QOK6."**  That closed 2021 Ascend report says only "the value differs from
  PyTorch" with screenshots and no numbers.  The observable here is narrower and reproducible on CPU
  in 2.9.0 with exact per-size values; if a maintainer considers it the same, this report still
  supplies the missing reproducer, the `n = 1` isolation and the controls.
