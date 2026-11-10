# Dataset

The benchmark uses a frozen 6,000-record subset of AG News, SST-2, and Emotion.
Each task contains 1,000 training records, 200 validation records, and 800 test
records. The sampling seed is 1234.

The repository tracks the source revisions, the complete manifest, class counts,
and split hashes. The text-bearing CSV is not tracked because the Emotion source
contains public social-media text whose redistribution terms must be confirmed.

Place `assignment4_real_6000.csv` in `data/processed/`. The validator also accepts
the supplied `assignment4_real_dataset_package.zip` directly:

```bash
python scripts/validate_dataset.py --dataset path/to/assignment4_real_dataset_package.zip
```

The expected CSV SHA-256 is
`75da6e2427b6ebc2418422f5b95bb47442f8d373d98697dd8f52159fe574cb82`.

