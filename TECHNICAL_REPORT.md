# A Policy-Grounded Decision Model for Synthetic Legal-Attention Routing

Dan Velton  
October 4, 2026

## Abstract

This project tests whether a small local model can make one constrained routing decision: whether a
fictional corporate request requires human lawyer attention. The model uses no real legal data.
Labels come from a versioned executable policy applied to structured synthetic facts. A frozen Laya
encoder feeds three independently trained multi-task heads that predict policy prerequisites,
escalation triggers, scope, workflow family, and the final binary decision.

The evaluation design emphasizes false clearances, counterfactual behavior, paraphrase stability,
source-separated test generation, out-of-distribution detection, long-document aggregation, and
finite-sample statistical risk bounds. The project is an engineering and research demonstration,
not a validated legal triage system.

## 1. Research question

The project asks a narrower question than general legal classification:

> Can a small local model learn an explicit fictional escalation policy well enough to clear some
> synthetic requests while routing uncertainty and policy exceptions to human attention?

This framing makes failure measurable. A false clearance occurs when the deterministic policy
requires attention but the model returns `NO_HUMAN_LAWYER_ATTENTION`. Over-escalation reduces
automation coverage but retains human review.

## 2. Policy-grounded supervision

The Northstar policy defines one prerequisite and twelve escalation triggers. The generator first
creates structured facts, applies the deterministic policy, and only then renders fictional prose.
The label is therefore inspectable and reproducible.

This approach avoids weak supervision from another language model. It also has a major limitation:
the model learns the project policy and generator, not law or legal judgment.

## 3. Benchmark design

The 102,000-record benchmark contains 25,500 four-record scenarios. Each scenario combines a
safe/attention counterfactual pair with two fact-preserving renderings. Training and validation use
procedural generator A. Test uses independently worded generator B. Challenge uses behavioral
generator C.

The benchmark includes:

- 47,800 single-trigger counterfactual records;
- 44,000 multi-trigger records;
- 10,200 out-of-distribution records;
- ten balanced workflow families; and
- tagged tests for negation, misleading headings, quotations, temporal updates, urgency,
  indirect triggers, conflicting facts, long context, typo noise, and multiple triggers.

## 4. Model

The pinned Laya encoder produces a 1,024-dimensional pooled representation and remains frozen.
Three trainable policy heads share the same architecture:

- input layer normalization;
- a 256-unit GELU hidden layer;
- dropout of 0.15;
- multi-label policy-task outputs; and
- a ten-class workflow-family output.

Only the heads are trained. The three members differ in initialization and sample order.

## 5. Training objectives

The primary loss is weighted binary cross-entropy over policy prerequisites, triggers, scope, and
the final attention output. Workflow-family cross-entropy supplies an auxiliary domain signal.

Two behavioral objectives are added:

1. a margin-ranking loss requires an attention-required counterfactual to score higher on the final
   attention output and on each changed trigger; and
2. a consistency loss reduces probability differences between fact-preserving paraphrases.

The heads train for 30 epochs with AdamW at a learning rate of 0.0003. Pooled encoder features are
cached locally as float16; head weights are saved as float32 safetensors.

## 6. Conservative inference

The model calculates a clearance score as the weakest policy condition:

```text
minimum(
  prerequisite probabilities,
  1 - trigger probabilities,
  in-scope probability,
  1 - final-attention probability
)
```

For long documents, the inference layer uses overlapping token windows. Prerequisite and scope
probabilities use the minimum across windows. Trigger and attention probabilities use the maximum.
A document exceeding 32 windows is routed to human attention.

## 7. Distribution and disagreement gates

Training-set family centroids are calculated in normalized feature space. Validation determines an
accepted distance to the nearest centroid. Validation also determines an accepted maximum
probability disagreement among the three heads.

An input that exceeds either limit is routed to human attention before threshold comparison.

## 8. Finite-sample risk control

The calibration stage evaluates the complete serialized ensemble on 12,000 validation records. It
selects a global clearance threshold and predicted-family thresholds using an exact one-sided
Clopper-Pearson binomial upper confidence bound for the false-clearance rate.

The independent statistical unit is the scenario, not each paraphrase. A scenario is a
false-clearance event if either attention-required rendering clears. Coverage is still measured
across individual records.

The policy limits are:

- 0.1% global false-clearance upper bound;
- zero observed validation false clearances at the selected global threshold;
- 1% workflow-family false-clearance upper bound;
- 95% confidence; and
- at least 200 attention-required validation scenarios for a family threshold.

If the evidence cannot support a threshold, clearance is disabled. Zero observed errors is not
treated as proof of zero risk.

## 9. Evaluation protocol

The held-out evaluation reports:

- false clearances and attention recall;
- observed and upper-bounded false-clearance rates;
- clearance coverage and clearance error rate;
- counterfactual pair consistency;
- paraphrase invariance;
- safe-score monotonicity;
- out-of-distribution attention rate;
- workflow-family and generator results;
- behavioral-capability results; and
- routing reasons from safety gates and thresholds.

The final test and challenge splits use the fresh benchmark seed `20261004` and are not used for
training or threshold selection. An earlier diagnostic evaluation informed design changes and was
retired rather than reported as final held-out evidence.

## 10. Results

| Split | Records | False clears | Attention recall | Clearance coverage | Pair consistency | Paraphrase invariance | Monotonicity | OOD attention | False-clear upper bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 12,000 | 0 | 100% | 13.43% | Not measured | Not measured | Not measured | Not measured | 0.0998% |
| Test | 10,000 | 0 | 100% | 13.43% | 26.86% | 90.10% | 90.78% | 100% | 0.1198% |
| Challenge | 5,000 | 0 | 100% | 5.16% | 10.32% | 95.52% | 90.04% | 100% | 0.2394% |

The model cleared 1,343 test requests and 258 challenge requests without an observed false
clearance. Validation supports a 95% scenario-level upper false-clear bound of 0.0998%, below the
0.1% policy limit. Zero observed test and challenge errors correspond to upper bounds of 0.1198%
and 0.2394%, so those smaller held-out scenario populations do not independently certify the 0.1%
target.

Pair consistency is much lower than attention recall because the metric requires both members of a
safe/risky pair to be routed correctly. The model frequently escalates the safe member, especially
under challenge transformations. That reduces automation value but does not create a false
clearance.

## 11. Interpretation limits

The strongest objections to the project are valid:

1. Synthetic success may reflect generator regularities rather than transferable legal reasoning.
2. A deterministic policy omits ambiguity, missing facts, strategic judgment, privilege, and
   organizational context.
3. A statistical bound on one generated validation population does not guarantee performance on a
   new population.

The design responds by separating generators, testing counterfactuals and paraphrases, routing OOD
and disagreement to humans, and stating the statistical claim narrowly. Those controls improve the
experiment but do not make it a production legal system.

## 12. Reproducibility

The repository pins:

- `laya==0.3.24`;
- Laya model revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`;
- `stuntd==0.1.2`; and
- reviewed upstream source revisions in `NOTICE`.

The benchmark seed is `20261004`. Policy, training configuration, thresholds, model revision,
normalization values, centroids, and validation summary are recorded in committed artifacts.

## 13. Intended use

The project can support internal discussion about policy design, selective automation, evaluation
methods, and human-review controls. It must not be used with real matters or represented as legal
advice, a replacement for lawyers, or evidence that automated legal triage is safe.
