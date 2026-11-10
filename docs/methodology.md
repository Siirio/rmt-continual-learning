# Methodology

## Research design

The study is a quantitative controlled comparative benchmark. It holds the model,
data subsets, prompt, task order, LoRA budget, optimizer, and evaluation procedure
constant. The adaptation strategy is the independent variable.

The primary research question is:

> How do replay-based and regularization-based continual fine-tuning approaches
> compare with standard sequential LoRA fine-tuning in mitigating catastrophic
> forgetting in open-source language models under limited computational resources?

H1 predicts that replay and regularization will reduce forgetting compared with
standard sequential LoRA. H2 predicts that replay will improve retention with an
additional storage or training cost and will therefore produce a different
retention-resource profile from regularization.

## Model and tasks

The model is `Qwen/Qwen2.5-0.5B`. The fixed task order is AG News, SST-2, then
Emotion. Each task contributes 1,000 training items, 200 validation items, and 800
held-out test items. All conditions use the same records in the same roles.

The prompt contains the task instruction, valid label options, input text, and an
answer field. Input text is truncated to 96 tokenizer tokens. The complete sequence
is limited to 160 tokens. Training loss applies only to the answer label and the
end-of-sequence token.

## Conditions

Sequential LoRA updates one adapter on each task in order. Experience replay adds a
class-stratified sample equal to 10 percent of each completed task. Online EWC
estimates a diagonal empirical Fisher from 200 per-example gradients, normalizes it
to mean one, and accumulates it with gamma equal to one. Joint training uses all
3,000 training records for two epochs as a non-continual reference.

## Outcomes

The primary outcome is average forgetting, FGT. Secondary outcomes are final
average accuracy, learning accuracy, backward transfer, macro-F1, per-task
accuracy, training time, peak GPU memory, optimizer steps, and additional storage.

The study reports mean and sample standard deviation over three seeds. Paired
bootstrap intervals use the same test items for both methods. Exact McNemar tests
compare paired correctness on earlier tasks. Holm correction controls the family
of comparisons.

## Validity controls

The test set is used only for final stage-wise measurement. EWC strength is chosen
on validation data with seed 1000. The runner records failures instead of removing
them. Each run stores its configuration, seed, environment, stage results, resource
measurements, and predictions.

