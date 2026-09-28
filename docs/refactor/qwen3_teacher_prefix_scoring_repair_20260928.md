# Teacher prefix conditional-probability repair

The readiness scorer had a real multi-token measurement defect. It compared
the probability of target A in context A against logits obtained by forcing
**target B's history** in context B, then gathering target A's token IDs. Once
the targets diverge before their final token, that is not
`log P(target A | context B)`. The same problem affected the reported
alternative-target probability within a context.

The preregistered definition remains unchanged: compare the same complete
target under the two contexts. The repair in
`src/posttrain_circuits/cli/evaluate_teacher_readiness.py` supplies each target's
own autoregressive history under the other context. It needs at most four
forwards per pair. Single-token targets and targets sharing the entire
teacher-forced prefix reuse the original forwards and are unchanged.

A deterministic CPU model reproduces a sign reversal: the old scorer returns
approximately `-0.0540671`; correct conditioning gives
`log(0.36 / 0.32) = +0.1177831`. The five regression cases cover multi-token
and three-token targets, different context lengths, shared target histories
and single-token targets. The new suite first produced three failures and two
passes; after the repair, all five pass. The affected combined suite passed
41 tests, and an independent reviewer reran the five new tests successfully.

The original own-context target probabilities, top-1 accuracy, top-k coverage,
retained mass, probe construction and all eight numeric thresholds are
unchanged. The strict-positive causal validity requirement is also unchanged.
This file is outside the 52-file student execution-safety surface; those
hashes still match. This is not a new execution-class certification.

The stored v5 capability manifest has differing target histories in 24/64
first-rule pairs and 63/64 intermediate-conclusion pairs: 87/128 pairs, or
174 scoring sides, need correctly conditioned model forwards. The raw logits
required for the replacement forwards were not saved, so the old ledger cannot
provide corrected values by arithmetic alone. Preserve its original report as
historical evidence and rerun affected measurements when evaluating a teacher.

This defect does not change the retained v7 generation result, 42/256 accepted
responses and 7/32 covered prompts. It does not fix citation/rule errors, prove
teacher readiness, or justify weakening any gate. Task adaptation and actual
GPU acceptance remain separate work, described in
`sdsc_teacher_adaptation_20260928.md`.
