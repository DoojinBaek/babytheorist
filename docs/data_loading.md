# BabyTheorist Data Loading Guide

This guide is the recommended loading contract for model development.

## Summary

BabyTheorist benchmark data is served from HDF5 files through `BabyTheoristEnv` and `BabyTheoristGym`.
The `train_data` object returned by `BabyTheoristGym.interact()` is path-backed and
lazy; it does not load the whole split when it is created. The slow path is
reading full grid rows one at a time in Python.

Use batch or slab reads:

```python
for batch in train_data.iter_batches(batch_size=4096):
    model.update_batch(batch.x, batch.y, task_id=batch.task_id)
```

If the current train split fits in memory and the model wants dense arrays:

```python
x, y = train_data.as_arrays()
model.fit_arrays(x, y)
```

Avoid:

```python
xs = np.stack([ex.x for ex in train_data])
ys = np.stack([ex.y for ex in train_data])
```

That code performs one HDF5 row read per task and is much slower than a slab
read.

## BabyTheoristGym Train Split

`BabyTheoristGym` gives one train split after a failed zero-observation test. Official
suite files contain 250K train observations per level:

```text
0-train test -> 250K train -> same-level eval attempts, up to 5 submits
```

If a model keeps state across calls, process the train payload once:

```python
while gym.active:
    data, phase = gym.interact()

    if phase == "train":
        for batch in data.iter_batches(batch_size=4096):
            model.update_batch(batch.x, batch.y)

    else:
        preds = model.predict(data)
        gym.submit(preds)
```

Use `data.as_arrays()` or `data.iter_batches(...)`; do not materialize by
iterating row examples and stacking them manually.

## File Layout

New HDF5 exports store `inputs` and `outputs` as contiguous uncompressed int8
datasets. The arrays are small and this layout is much faster for both slab
reads and row reads than auto-chunked compressed spatial blocks.

You can inspect a file with:

```bash
python scripts/bench_data_loader.py --data-dir data --mode single_easy --level 1 --n 5000
```

In the sidecar metadata, new files include:

```json
{"h5_layout": {"inputs_outputs": "contiguous_uncompressed"}}
```

Older generated files without this field, or files whose `inputs` / `outputs`
datasets show non-`None` `chunks` or `compression`, should be regenerated or
rewritten before high-throughput model training.

## Eval Loading

During test phases, `eval_data` is also lazy and exposes one
`EvalExample` per task:

```python
for ex in eval_data:
    pred = model.predict_one(ex.support_x, ex.support_y, ex.query_x)
```

Return predictions as `(task_id, grid)` pairs and let `gym.submit(...)` or
`env.score_predictions(...)` read the official answer set. `BabyTheoristGym` starts
each level with a zero-observation entry test, then allows one charged eval
submission after training by default. The entry test can contribute to
the official per-level best score, but it does not consume the charged eval
budget. Do not call `env.answers(...)` inside model code except for
oracle/debugging experiments.
