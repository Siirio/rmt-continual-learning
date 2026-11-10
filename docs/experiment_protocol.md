# Experiment Protocol

## Frozen settings

| Setting | Value |
| --- | --- |
| Model | Qwen/Qwen2.5-0.5B |
| Task order | AG News, SST-2, Emotion |
| Records per task | 1,000 train, 200 validation, 800 test |
| Data seed | 1234 |
| Main seeds | 0, 1, 2 |
| LoRA | rank 8, alpha 16, dropout 0.05 |
| LoRA targets | q, k, v, and o projections |
| Optimizer | AdamW, learning rate 2e-4, weight decay 0 |
| Training | 2 epochs, effective batch 16, 10% warm-up |
| Gradient clipping | 1.0 |
| Sequence limits | 160 total tokens, 96 text tokens |
| Replay | 10% per completed task, class-stratified |
| EWC Fisher | 200 per-example gradients, mean-one normalization |
| EWC grid | 0.01, 0.1, 1, 10, 100 |
| EWC calibration seed | 1000 |

The frozen dataset CSV SHA-256 is
`75da6e2427b6ebc2418422f5b95bb47442f8d373d98697dd8f52159fe574cb82`.
The nine split hashes are stored in
`data/manifests/assignment4_split_hashes.csv`.

## Run schedule

The executable schedule contains 19 runs:

- 3 Sequential LoRA main runs
- 3 experience replay main runs
- 3 online EWC main runs
- 3 joint-training reference runs
- 5 validation-only EWC calibration runs
- 2 replay-buffer sensitivity runs at 5% and 20%

The earlier Assignment 3 appendix described 17 runs. The Assignment 4 repository
request adds the two replay sensitivity runs. This is a declared extension. It does
not change the main comparison, datasets, seeds, or hypotheses.

## Support criteria

H1 is supported for a method when its FGT is lower than Sequential LoRA and the
paired retention interval excludes zero on earlier tasks. H1 is not supported when
retention does not improve or when lower FGT results from failure to learn the new
task.

H2 is supported when replay improves retention and records a measurable increase in
training work or stored replay data. H2 is not supported when replay produces no
retention gain or no distinct resource profile.

## Output contract

Every planned run produces a JSON file. Completed files contain the result and
environment. Failed files contain the exception type, message, traceback, and run
specification. Resume mode skips an existing run file. No failed run is silently
removed.

