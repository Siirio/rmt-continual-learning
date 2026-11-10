# Reproducibility

## Environment

Use Python 3.10 or newer. A CUDA-capable environment is required only for the full
model experiment. Ordinary validation, statistics, result generation, and tests run
on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiment,reporting,dev]"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Data placement

Place the frozen CSV at `data/processed/assignment4_real_6000.csv`. Confirm that its
SHA-256 matches the manifest. The validator can also inspect the supplied ZIP
package without extraction.

```bash
python scripts/validate_dataset.py --dataset path/to/assignment4_real_dataset_package.zip
```

## Planned runs

List the exact schedule before execution:

```bash
rmt schedule
```

Run one experiment:

```bash
python scripts/run_experiment.py --run-id sequential_lora_seed0
```

Run the full schedule with resume behavior:

```bash
python scripts/run_experiment.py --full
```

Each completed or failed run is written immediately to `results/raw/`. Repeating the
same command skips existing files. Use `--no-resume` to replace this behavior only
when a documented rerun is required.

## Analysis and figures

```bash
python scripts/analyze_results.py
python scripts/generate_figures.py --input results/tables --output results/figures
```

## Checkpoint recovery

`CheckpointManager` stores trainable adapter parameters and stage metadata under a
run identifier. Load the checkpoint into the same model and LoRA configuration,
then continue from the recorded stage. A parameter-name mismatch stops the restore
instead of applying a partial state.

## Determinism record

The run files record Python and platform information, seeds, method parameters, and
elapsed runner time. The full environment should also record GPU model, CUDA,
PyTorch, Transformers, PEFT, and Datasets versions before measured execution.

