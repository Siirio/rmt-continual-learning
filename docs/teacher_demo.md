# Teacher Demonstration

This sequence fits a 5 to 7 minute demonstration.

## 1. Research question

Open `README.md`. Explain that the study compares replay and regularization with a
standard sequential LoRA baseline under the same model, data, and compute setting.

## 2. Frozen dataset

Open `data/manifests/assignment4_dataset_manifest.json` and
`data/manifests/validation_report.json`. Show the 6,000-record count, source
revisions, CSV hash, and 1,000/200/800 split for each task.

Run:

```bash
python scripts/validate_dataset.py --dataset path/to/assignment4_real_dataset_package.zip
```

## 3. Training methods

Open these files in order:

- `src/rmt/methods/sequential_lora.py`
- `src/rmt/methods/experience_replay.py`
- `src/rmt/methods/ewc.py`
- `src/rmt/methods/joint_training.py`

Point out the shared `run` interface, the stratified replay allocation, the Fisher
normalization, and the joint reference.

## 4. Forgetting measurement

Open `src/rmt/evaluation/metrics.py`. Explain the accuracy matrix, ACC, LA, BWT, and
FGT. Then run:

```bash
python -m pytest tests/test_metrics.py tests/test_statistics.py
```

## 5. Run schedule and recovery

Run:

```bash
rmt schedule
```

Show that the output contains 19 unique run identifiers. Open
`src/rmt/training/runner.py` to show immediate result persistence, failure records,
and resume behavior.

## 6. Tables and figures

Run:

```bash
python scripts/generate_examples.py
python scripts/generate_figures.py --input examples/results --output examples/figures
```

Open `examples/results/manifest.json` to show the example-data marker and generator
seed. Open the accuracy, forgetting, retention, and cost figures.

## 7. Reproducibility close

Show `configs/`, `docs/reproducibility.md`, and the Colab notebook. Explain that the
full model command exists, but the repository checks were completed without starting
the full model experiment.

