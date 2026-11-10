# Results Reference

## Directory roles

- `results/raw/` stores future model-run JSON files
- `results/tables/` stores derived tables from measured runs
- `results/figures/` stores derived figures from measured runs
- `examples/results/` stores deterministic illustrative files
- `examples/figures/` stores figures derived only from the example package

## Example schemas

`per_seed_metrics.csv` contains one row for each method and seed. It records ACC,
macro-F1, LA, BWT, FGT, and final task accuracies.

`accuracy_matrices.json` maps method and seed to a four-by-three matrix. Row zero is
the base model. The remaining rows follow training on T1, T2, and T3.

`main_comparison.csv` contains means, sample standard deviations, resource means,
and additional storage. Joint-training LA, BWT, and FGT fields are empty because
those sequential metrics do not apply.

`statistical_comparison.csv` stores paired accuracy differences, bootstrap interval
bounds, McNemar discordant counts, raw p-values, and Holm-adjusted p-values.

Each file in `runs/` contains metadata, the accuracy matrix, metrics, resources, and
per-item gold labels, predictions, and correctness flags. The metadata field
`example_data` is true.

## Regeneration

```bash
python scripts/generate_examples.py
python scripts/generate_figures.py --input examples/results --output examples/figures
```

The generator seed is 20261110. Running the commands again with the same source
version produces the same numeric package.

