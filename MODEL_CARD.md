# Model Card: Legal Decision Model

## Model details

- Name: Legal Decision Model
- Version: 0.1.0
- Date: October 2, 2026
- Developer: Dan Velton
- Task: binary legal-attention routing
- Base model: `convaiinnovations/laya` 0.3.24
- Task adaptation: frozen Laya encoder with a trained `stuntd` decision head
- License: Apache License 2.0
- Training data: generated fictional requests only

The two output labels are:

- `REQUIRES_HUMAN_LAWYER_ATTENTION`
- `NO_HUMAN_LAWYER_ATTENTION`

The external decision is conservative. The model returns `NO_HUMAN_LAWYER_ATTENTION` only when the
calibrated probability for that label meets the selected clearance threshold. A lower-confidence
answer is routed to human attention.

Calibration is fitted after the trained head is serialized to float16 and reloaded through the
serving path. Inputs that would be truncated by the model's token budget bypass model inference and
are routed to human attention.

## Intended use

This model is an educational proof of concept for:

- testing a narrow decision-model architecture
- demonstrating deterministic synthetic labeling
- studying false-clearance and selective-classification metrics
- prototyping local legal-operations routing interfaces

Any real deployment would require a separately approved policy, representative authorized data,
privacy and security review, legal review, governance, monitored shadow testing, and a documented
human override.

## Uses outside scope

Do not use the model:

- to give legal advice or interpret law
- to decide legal rights, obligations, liability, or strategy
- to clear a real matter without an approved human-governed process
- for employment, housing, credit, insurance, healthcare, immigration, criminal justice, or other
  high-impact decisions
- with privileged, confidential, personal, customer, employee, or production data
- as evidence that automated legal triage is accurate or safe

## Training

The task head is trained for 24 epochs on 7,500 generated records. Laya's encoder remains frozen.
Encoder outputs are cached, and only the decision head is updated. A 1,000-record validation split
is used to:

1. fit a temperature for probability calibration;
2. select the lowest safe-clearance threshold that meets the configured false-clear limit; and
3. maximize correct safe clearances subject to that limit.

The default configured validation false-clear limit is zero. This controls the observed synthetic
validation split only.

## Evaluation

The evaluation suite reports:

- accuracy
- attention recall
- false-clear rate
- clearance error rate
- clearance coverage
- safe-clearance recall
- Brier score
- counterfactual-pair consistency
- per-family metrics

Measured values are written to `evaluation/current/test.json` and
`evaluation/current/challenge.json` after training. The earlier 3-epoch, 3,250-record experiment is
preserved under `evaluation/baseline-3-epoch/` for comparison.

| Split | Records | Accuracy | Attention recall | False-clear rate | Clearance coverage | Pair consistency |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 1,000 | 74.2% | 100% | 0% | 24.2% | Not measured |
| Test | 1,000 | 81.5% | 100% | 0% | 31.5% | 63.0% |
| Challenge | 500 | 62.6% | 100% | 0% | 12.6% | 25.2% |

Most errors are over-escalations: requests labeled safe by the fictional policy that the model
routes to a lawyer. The challenge results show that the model is highly conservative under wording
and format shifts. The observed zero false-clear rate applies only to these generated datasets.

## Data

No real legal, customer, employee, contract, privileged, or internal company data is used. The
dataset is generated from explicit fictional facts and a deterministic fictional policy. See
[DATASET_CARD.md](DATASET_CARD.md).

## Limitations

The examples use limited templates, vocabulary, and fact combinations. Counterfactual pairs improve
the test design but do not reproduce the ambiguity of real legal work. Calibration can degrade
under wording, policy, or population changes. The model does not produce a legal rationale and
cannot verify whether a requester supplied complete or truthful facts.

The model has a finite input token budget. The implementation detects that condition using the
exact tokenizer, question layout, and saved sequence limits. It does not clear a request if any
input tokens would be discarded.

The base checkpoint and training libraries are early-stage software. Upstream behavior and
interfaces may change. This repository pins package versions and records source revisions in
[NOTICE](NOTICE), but reproducibility still depends on the Python, PyTorch, operating-system, and
hardware environment.
