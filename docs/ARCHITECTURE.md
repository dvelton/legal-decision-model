# Architecture

## Decision contract

The model has one external decision:

- `REQUIRES_HUMAN_LAWYER_ATTENTION`
- `NO_HUMAN_LAWYER_ATTENTION`

Internally, the model predicts the policy facts that support that decision. A request is cleared
only when every required condition passes the applicable statistical and model-safety gates.

## Data flow

```text
versioned YAML policy
  -> deterministic structured facts and labels
  -> source-separated synthetic benchmark
  -> frozen Laya encoder
  -> pooled 1,024-dimensional representation
  -> three independently trained policy heads
  -> policy-task, scope, final-decision, and workflow-family outputs
  -> conservative window aggregation
  -> OOD and ensemble-disagreement gates
  -> global and workflow-family risk thresholds
  -> binary routing decision
```

## Policy compiler

`policies/northstar.yaml` defines:

- the two external labels;
- prerequisites for self-service clearance;
- escalation triggers and their human-readable reasons;
- the maximum false-clearance rates;
- the ensemble and long-document settings; and
- the training hyperparameters.

`policy_schema.py` validates the file and compiles it into a typed `PolicySpec`. Generation,
training, calibration, evaluation, and inference all load the same policy contract.

## Benchmark

The generated benchmark contains 102,000 fictional records:

| Split | Records | Generator |
|---|---:|---|
| Train | 75,000 | Procedural generator A |
| Validation | 12,000 | Held-out generator A |
| Test | 10,000 | Independently worded generator B |
| Challenge | 5,000 | Adversarial behavioral generator C |

Each scenario produces four records: a safe request and its attention-required counterfactual,
each expressed in two different renderings. The design supports three separate tests:

1. whether the model changes its answer when a policy fact changes;
2. whether the model preserves its answer when wording changes but facts do not; and
3. whether the model remains conservative under unfamiliar or misleading presentation.

The benchmark is generated locally under `benchmark/` and is excluded from Git because the full
JSONL files are large. Generation is deterministic from the committed policy, code, and seed.

## Laya feature encoder

The project uses only the pinned Laya model family. The encoder remains frozen. For each input:

1. the pinned tokenizer creates model tokens;
2. the frozen encoder produces token representations;
3. non-padding token representations are mean-pooled; and
4. the resulting vector is cached as float16.

The cached training matrix is about 150 MB for 75,000 rows at 1,024 dimensions. This makes repeated
head experiments practical without repeatedly running the full encoder.

## Multi-task ensemble

Each of the three policy heads has:

- an input layer normalization;
- a 256-unit hidden layer with GELU and dropout;
- one output for every policy prerequisite and trigger;
- an `in_scope` output;
- a final attention-decision output; and
- a ten-class workflow-family output.

The heads use different random initializations and batch orders. They share the frozen Laya
features but do not share trainable weights.

Training combines:

- binary cross-entropy for policy tasks;
- cross-entropy for workflow family;
- margin ranking for safe/attention counterfactual pairs; and
- consistency loss for fact-preserving paraphrases.

## Clearance score

The model does not clear a request based only on its final binary output. It calculates a
conservative clearance score as the minimum of:

- every prerequisite probability;
- one minus every escalation-trigger probability;
- the in-scope probability; and
- one minus the final attention probability.

One weak policy signal therefore lowers the entire request's clearance score.

## Long documents

Inputs longer than one model window are divided into overlapping token windows. The system:

- takes the minimum prerequisite and in-scope probabilities across windows;
- takes the maximum trigger and attention probabilities across windows;
- uses the maximum OOD distance across windows; and
- refuses clearance if the document requires more than the configured maximum number of windows.

A risky sentence cannot be made irrelevant merely by surrounding it with enough safe text.

## OOD and disagreement controls

The validation set determines:

- a maximum accepted distance to the nearest workflow-family centroid; and
- a maximum accepted disagreement among ensemble members.

Either condition routes the request to human attention before threshold comparison.

## Statistical risk control

Threshold selection uses an exact one-sided Clopper-Pearson binomial upper confidence bound for the
false-clearance rate. The selected threshold maximizes correct clearances while keeping the upper
confidence bound below the policy limit. A global threshold and separate predicted-family
thresholds are fitted.

If no threshold has enough supporting validation evidence, clearance is disabled. This is
finite-sample binomial risk control. It is not a claim of zero risk and is not described as full
conformal risk control.

## Artifact layout

```text
policy-model/
  head-0.safetensors
  head-1.safetensors
  head-2.safetensors
  metadata.json
  risk-control.json
```

The metadata records the policy and base-model revisions, task ordering, feature normalization,
family centroids, thresholds, window configuration, training size, and validation summary.

## Compatibility

The public v0.1 artifact remains under `model/`. Its single `stuntd` head and measured evaluations
are preserved for reproducibility. The v2 artifact uses a separate `policy-model/` directory and
format version, so the original model is not overwritten.
