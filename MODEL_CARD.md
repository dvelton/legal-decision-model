# Model Card: Legal Decision Model Policy Ensemble

## Model details

- Name: Legal Decision Model
- Version: 0.2.0
- Date: October 4, 2026
- Developer: Dan Velton
- Task: binary fictional legal-attention routing
- Base model: `convaiinnovations/laya`
- Base revision: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
- Adaptation: frozen Laya encoder with three independently trained multi-task policy heads
- License: Apache License 2.0
- Training data: generated fictional requests only

The external labels are:

- `REQUIRES_HUMAN_LAWYER_ATTENTION`
- `NO_HUMAN_LAWYER_ATTENTION`

## Intended use

The model is an educational and research proof of concept for:

- testing a narrow policy-routing architecture;
- studying selective automation and human escalation;
- demonstrating deterministic synthetic supervision;
- evaluating counterfactual and paraphrase behavior;
- testing finite-sample risk controls; and
- prototyping local legal-operations interfaces without real legal data.

Any operational experiment would require a separately approved policy, authorized representative
data, privacy and security review, legal review, governance, monitored shadow testing, documented
human override, incident handling, and continuing performance review.

## Uses outside scope

Do not use the model:

- to give legal advice or interpret law;
- to decide legal rights, obligations, liability, or strategy;
- to clear a real matter without an approved human-governed process;
- for employment, housing, credit, insurance, healthcare, immigration, criminal justice, or other
  high-impact decisions;
- with privileged, confidential, personal, customer, employee, or production data;
- as evidence that automated legal triage is accurate or safe; or
- to bypass an existing legal, privacy, security, compliance, employment, or regulatory review.

## Training data

The model trains on 75,000 records from the 102,000-record Northstar Legal Attention Benchmark.
A separate 12,000-record validation split controls thresholds. Test and challenge records are held
out.

Every label comes from the deterministic Northstar 2.0 policy. No language model chooses labels.
See [DATASET_CARD.md](DATASET_CARD.md).

## Architecture

The frozen Laya encoder produces a pooled 1,024-dimensional feature vector. Each policy head
contains:

- input layer normalization;
- a 256-unit GELU hidden layer;
- dropout of 0.15;
- outputs for every policy prerequisite and escalation trigger;
- an in-scope output;
- a final attention output; and
- a ten-class workflow-family output.

Three heads use different initializations and batch orders. The inference system averages their
task probabilities and checks their disagreement.

## Training

Each head trains for 30 epochs with AdamW and a learning rate of 0.0003. The objectives are:

1. weighted binary cross-entropy for policy tasks;
2. workflow-family cross-entropy;
3. counterfactual margin ranking; and
4. fact-preserving paraphrase consistency.

The encoder remains frozen. Pooled features are cached as float16. Head weights are stored as
float32 safetensors.

## Conservative inference

The model's clearance score is the minimum of every condition needed for self-service:

- prerequisite probabilities;
- inverted escalation-trigger probabilities;
- in-scope probability; and
- inverted final-attention probability.

For long inputs, overlapping windows are aggregated conservatively. Any risky window can lower the
clearance score. Inputs exceeding the maximum window count route to human attention.

The model also routes to human attention when:

- embedding distance exceeds the validation OOD limit;
- ensemble disagreement exceeds the validation limit;
- calibration lacks a global or predicted-family threshold; or
- the clearance score falls below either applicable threshold.

## Statistical claim

Threshold selection uses an exact one-sided Clopper-Pearson binomial upper confidence bound. The
Northstar policy requires:

- a maximum 0.1% global false-clearance upper bound;
- zero observed validation false clearances at the selected global threshold;
- a maximum 1% workflow-family false-clearance upper bound;
- 95% confidence; and
- at least 200 attention-required scenarios for a family threshold.

The statistical trial is the independently generated scenario. Two attention-required paraphrases
from one scenario count as one trial, with any false clearance making that scenario an error.
Coverage remains record-level.

The claim is limited to the generated validation population and selected operating point. It is not
a guarantee of future performance and is not full conformal risk control.

## Evaluation

The v2 suite reports:

- attention recall;
- false clearances;
- observed and upper-bounded false-clearance rates;
- clearance coverage and clearance error rate;
- counterfactual pair consistency;
- paraphrase invariance;
- safe-score monotonicity;
- out-of-distribution attention rate;
- per-family, per-generator, and per-capability metrics; and
- routing-reason counts.

Measured v2 reports are stored under `evaluation/policy-v2/`. The v0.1 reports remain under
`evaluation/current/` and `evaluation/baseline-3-epoch/`.

### Measured results

| Split | Records | False clears | Attention recall | Clearance coverage | Pair consistency | False-clear upper bound |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 12,000 | 0 | 100% | 13.43% | Not measured | 0.0998% |
| Test | 10,000 | 0 | 100% | 13.43% | 26.86% | 0.1198% |
| Challenge | 5,000 | 0 | 100% | 5.16% | 10.32% | 0.2394% |

Validation satisfies the configured 0.1% upper-bound target. Test and challenge each have zero
observed false-clearance scenarios, but their scenario-level upper bounds are 0.1198% and 0.2394%.
Pair consistency is limited by conservative treatment of safe examples; attention recall remained
100% on both held-out splits.

## Limitations

The model does not learn law or legal judgment. It learns a fictional policy expressed through a
synthetic generator. Source separation, counterfactuals, paraphrases, and adversarial
transformations improve the experiment but cannot reproduce:

- incomplete or strategically framed real intake;
- privilege and confidentiality concerns;
- organizational context and risk tolerance;
- legal ambiguity and changing law;
- interactions among policies, jurisdictions, and business facts; or
- distribution shifts in operational use.

The model cannot verify that supplied facts are complete or true. Predicted policy reasons are
diagnostic outputs, not legal conclusions.

The base model and training libraries are early-stage software. The repository pins package and
model revisions, but reproducibility still depends on Python, PyTorch, operating system, hardware,
and upstream artifact availability.

## Preserved v0.1 model

The repository retains the original single `stuntd` head under `model/`. It was trained for 24
epochs on 7,500 synthetic records and calibrated on 1,000 validation records. Its observed results
are preserved for historical comparison, not presented as evidence about the v2 ensemble or real
legal requests.
