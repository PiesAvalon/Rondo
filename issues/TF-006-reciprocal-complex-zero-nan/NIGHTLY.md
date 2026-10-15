# TF-006 Nightly verification — 2026-09-27

The tensor-length inconsistency remains in TensorFlow Nightly, but the vectorized
result has changed from `nan+nanj` to `inf-infj`. The exact historical
`nan+nanj` symptom was not reproduced on this build.

| Environment | Value |
| --- | --- |
| TensorFlow runtime version | `2.23.0-dev20260925` |
| Installed distribution | `tf-nightly==2.23.0.dev20260925` |
| Git version | `v1.12.1-145528-g4470f2f1dd9` |
| Keras Nightly | `3.16.0.dev2026092604` |
| Python / NumPy | `3.12.13` / `2.5.1` |
| OS / device | Ubuntu 24.04.4 LTS, WSL2, x86_64 / CPU |
| Fresh-process replays | 2/2 reproduced the component-class mismatch |

The existing TensorFlow environment supplies Python and the shared dependencies.
Only TensorFlow Nightly and Keras Nightly were installed in
`/home/tommy_ge/.cache/rondo/tf-nightly-2.23.0.dev20260925` and loaded through
`PYTHONPATH`. Package requirements were checked, and both the TensorFlow Python
module and native extension were verified to come from that directory.
The existing TensorFlow 2.21.0 installation was not modified.

For `complex64`, the minimal example from the issue form now produces:

```text
3 [(inf+nanj), (inf+nanj), (inf+nanj)]
4 [(inf-infj), (inf-infj), (inf-infj), (inf-infj)]
5 [(inf-infj), (inf-infj), (inf-infj), (inf-infj), (inf+nanj)]
```

`complex128` shows the same split with groups of two elements. In both replays,
the component comparison covered both dtypes at lengths 1, 2, 3, 4, 5, 8, and 9:
10 eager cases differed from NumPy, while all 14 XLA cases matched NumPy's
`inf+nanj` result. The nonzero control `1/(1+0.5j)` matched NumPy in both modes.

The old `repro.py` returns seven failed checks per replay because it explicitly
expects the old `nan+nanj` result. Its XLA checks also only exclude `nan+nanj`.
The new `repro_nightly.py` compares the real and imaginary component classes
against NumPy, including the signs of infinities.

Run from this directory:

```bash
PYTHONPATH=/home/tommy_ge/.cache/rondo/tf-nightly-2.23.0.dev20260925 \
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 \
/mnt/e/2026-2027/Rondo/.rondo-runtime/tensorflow/bin/python repro_nightly.py
```

Expected diagnostic summary on this Nightly build:

```text
RESULT: mismatching cases: {'eager': 10, 'XLA': 0}
```

Exit code 1 means a mismatch was observed. Exit code 0 means all cases matched.
See [environment and original replays](nightly-2026-09-27.txt) and
[component comparison replays](nightly-2026-09-27-components.txt).

## Updated issue-form fields

**Have you reproduced the bug with TensorFlow Nightly?**

Yes — the tensor-length inconsistency remains, with a different vectorized result.

**TensorFlow version**

`2.23.0-dev20260925` (tf-nightly)

**Current behavior?**

On CPU in eager mode, `tf.math.reciprocal` returns different results for identical
complex zeros depending on tensor length. For `complex64`, length 3 returns
`inf+nanj` for every element, length 4 returns `inf-infj` for every element, and
length 5 returns four `inf-infj` values followed by one `inf+nanj`. `complex128`
shows the same pattern with groups of two elements.

Expected: every element should return `inf+nanj`, matching NumPy and TensorFlow
XLA, regardless of tensor length.

This inconsistency also occurs in TensorFlow 2.21.0, where the vectorized result
is `nan+nanj` instead. The Nightly behavior was reproduced in two fresh processes.
