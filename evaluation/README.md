# Evaluation Results

`baseline-3-epoch/` preserves the first experiment:

- 3,250 generated records
- 2,400 training records
- 300 validation records
- 3 head-training epochs

`current/` contains the selected 24-epoch experiment on the expanded 10,000-record dataset.

Each directory contains:

- `model.json`: calibration and validation metadata
- `test.json`: held-out ordinary test results
- `challenge.json`: held-out negation and unfamiliar-phrasing results

The experiments use different generated datasets, so the comparison measures the combined effect
of the larger, harder dataset and longer task-head training. It does not isolate the effect of
epochs alone.
