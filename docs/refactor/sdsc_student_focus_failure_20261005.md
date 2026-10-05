# V5 preparation: complete execution, rejected model

Job54663831 completed its planned32 full-parameter updates,1,665,064 model-input
tokens and all18,432 development responses. Fresh Expanse accounting at
2026-10-05T05:56:09Z reports job/batchFAILED1:0, externCOMPLETED0:0,21200 seconds
and an empty queue. The original selector returns no eligible checkpoint.
This is a completed quality rejection, not a demonstrated execution crash.

## Measurements under the unchanged gates

All table entries are complete-proof counts out of256. IID answer counts equal
IID proof counts at these steps; transformed answer counts are not substituted.

| Step | IID | Rename | Fact order | Rule order | Paraphrase | Distractors | Rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 57 | 49 | 49 | 46 | 59 | 46 | Branch3/68 and DAG0/102 below structure floors |
| 6 | 97 | 75 | 74 | 71 | 84 | 63 | Each view gap exceeds5pp |
| 7 | 131 | 97 | 116 | 112 | 122 | 94 | Four view gaps and mean gap |
| 8 | 153 | 132 | 143 | 130 | 160 | 124 | Rename8.20pp, rule8.98pp, distractor11.33pp, mean5.94pp |
| 12 | 191 | 181 | 191 | 177 | 195 | 179 | IID above153 and rule gap5.47pp |
| 16 | 221 | 195 | 224 | 213 | 223 | 214 | IID ceiling and rename gap |
| 24 | 245 | 210 | 244 | 236 | 240 | 237 | IID ceiling and rename gap |
| 32 | 251 | 221 | 245 | 242 | 255 | 243 | IID ceiling and rename gap |

Step5 meets the view-gap criteria without adequate multi-branch capability.
Step6 learns both branch and DAG proofs, but IID success increases faster than
transformed success. Step8 is exactly at the153-answer upper bound and still
misses three individual robustness gates. Removing the upper bound alone would
not admit any observed later checkpoint: each still fails at least one view gap.
No checkpoint from this run can proceed to formal qualification.

## What the failures show

An independent explanatory replay reproduces the original parser/verifier
outcomes for every response. At step8, rename loses54 IID-success cases while
gaining33;46 of the54 losses are premise mismatches and six are malformed
length-terminated outputs. Branches account for17 of the net21-proof deficit.
For distractors, the net29-proof deficit includes16 DAG,7 chain and6 branch
cases; only3 of the45 IID-success losses reach the token cap. Branch sampling
and formatting alone do not explain all remaining failures.

Step2 has a separate temporary decoding failure:1533/1536 outputs repeat and
reach the256-token cap. Step3 recovers to67 length terminations. This supports
testing the update size, but does not prove LR is the unique cause. Loss values
come from changing teacher-forced batches and cannot establish generative quality.
The old5e-4 optimization overshoot and this5e-5 quality rejection are different
observations. There is no new evidence of incorrect AdamW arithmetic.

The independent execution review verifies64 finite, nonzero rank-updates,
24 exact FP32 reloads,48 native-BF16 parity checks,24 finite2454-token probes
and actual two-rank step4 full-state restoration. All680 own-job memory samples
pass, with peak166.94170GiB and217.05830GiB headroom; no observed OOM/failcnt.
Nineteen large files totaling119,417,322,624 bytes have persistent read-back
evidence. Publication took375.864 seconds within the600-second reserve. Both
ranks deliberately return2 for null selection; launcher termination follows
this result. GPU arithmetic and large checkpoint contents were not independently
recomputed on Quest.

## Evidence and continuation

The immutable fit plan/receipt remains under
`.sdsc/student-focus-v5/8da3db1571a34957e9e03992ebd16f73/`; source is accepted
71d0508407d429e5f700a3908671af1244b63f1f. Small fetch139,520,092 bytes is
`.sdsc/fetched/54663831/fetch-k0xc6f4b/`; weights remain on Expanse.

- Publication: `e2e2f8aba7d45d6c0adfef13e8f9af60873f648e16f18a51a3aae91399ce02d0`.
- Report: `36c6c539823d6d602f3955ab9a1cebe8fe4b5bc0e699972ce6b2dba1be8bd069`.
- Frozen raw audit: `508004bc7d940409afa6b9d10c915ee95644a0070f41fc0a51cf728df6a529e3`.
- Independent execution review: `b9ee3f3348db5b4f541438460cce16c6e308b84d64432c029ed908b60c08d3d8`.
- Independent result-binding review: `74774fda089b92f0e4f19d8401a05c2a4a18129753f8cc31578f39c825d6497e`.
- Explanatory replay: `7d2d361c7f59e316924b574523ba4143fccb5228a4a5502ea91b53106dbb90f4`.

Audits and fresh status/fetch records are under
`.sdsc/diagnostics/student-focus-v5/`. The old observer remains terminal after
SSH loss; reconnection used the actualquser32 master, not the oldquser43 path.
No existing job was resubmitted and no formal qualification was started.

The next prospective exploration changes only constant LR,5e-5 versus2.5e-5,
with identical newly generated data and complete32-update exposure. It retains
V5 sampling, original proof semantics and all original numerical gates.
Seven checkpoints2/4/6/8/12/16/32 measure the early decoding transient and later
capability/robustness trajectories; no diagnostic checkpoint can be promoted.
Protocol, independent review and actual submission are separate next steps.
Repeated developer-set adaptation remains exploratory even with new seeds;
neither a smaller gap caused solely by worse IID performance nor a shifted
learning curve establishes a broad robustness repair.
