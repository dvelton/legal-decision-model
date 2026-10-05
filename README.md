# Legal Decision Model

Legal Decision Model is a local proof of concept with one job: read a fictional corporate request and decide whether it requires human lawyer attention.

The project generates its own synthetic data, trains a small decision head on the Apache-licensed Laya encoder, and applies a conservative clearance threshold. A request receives NO_HUMAN_LAWYER_ATTENTION only when the model is sufficiently confident. Everything else receives REQUIRES_HUMAN_LAWYER_ATTENTION.

This is a demonstration, not legal advice. It has not been tested on real legal matters and should not be used to bypass legal, compliance, privacy, security, employment, or regulatory review.

## What the project demonstrates

- A versioned YAML policy defines the conditions for self-service and escalation.
- A deterministic generator creates a 102,000-record synthetic benchmark without real legal data.
- A frozen Apache-licensed Laya encoder runs locally.
- Three small policy heads learn the policy facts, scope, workflow family, and final decision.
- Counterfactual ranking teaches the model to respond when a policy-relevant fact changes.
- Paraphrase consistency teaches the model to preserve decisions when wording changes but facts do
  not.
- Long documents are evaluated through overlapping windows rather than silent truncation.
- Out-of-distribution inputs and ensemble disagreement route to human attention.
- Finite-sample statistical bounds determine whether any safe clearance is permitted.

The original v0.1 single-head model remains under `model/`. The improved policy ensemble uses the
separate `policy-model/` artifact format.

## Try the included model

Requirements:

- macOS, Linux, or Windows
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- about 5 GB of free disk space for dependencies and the Laya checkpoint

```bash
git clone https://github.com/dvelton/legal-decision-model.git
cd legal-decision-model
uv sync
uv run legal-decision-model policy validate
uv run legal-decision-model policy decide --text \
  "The counterparty changed the approved liability language."
```

The first model command downloads the pinned Laya checkpoint.

Run the local browser interface:

```bash
uv run legal-decision-model serve
```

Open `http://127.0.0.1:8000`. The server prefers the v2 policy ensemble when its artifact is
installed and otherwise loads the preserved v0.1 model.

## Output

The v2 decision includes:

- the binary routing answer;
- a conservative clearance score and required threshold;
- predicted workflow family;
- policy-task probabilities;
- routing reasons;
- ensemble disagreement;
- distance from the learned workflow distributions;
- token and window counts;
- model and policy provenance; and
- an explicit synthetic-data disclaimer.

The internal signals make the result inspectable. They are model predictions, not legal findings.

## Generate the benchmark

The full JSONL benchmark is reproducible but excluded from Git because of its size:

```bash
uv run legal-decision-model policy generate --output benchmark --seed 20261004
uv run legal-decision-model policy validate-benchmark --data benchmark
```

| Split | Records | Purpose |
|---|---:|---|
| Train | 75,000 | Fit the three policy heads |
| Validation | 12,000 | Set OOD, disagreement, and risk thresholds |
| Test | 10,000 | Test independently worded generation |
| Challenge | 5,000 | Stress behavioral and adversarial transformations |

Each of the 25,500 scenarios contains a safe/attention counterfactual pair in two fact-preserving
renderings. See [DATASET_CARD.md](DATASET_CARD.md).

## Train the improved model

Apple Silicon:

```bash
uv run legal-decision-model policy train \
  --data benchmark \
  --model-dir policy-model \
  --device mps
```

Use `--device cpu` on a machine without Apple Silicon or CUDA. Pooled Laya features are cached under
`artifacts/policy-feature-cache/`, so a repeated head-training run does not rerun the full encoder
unless the data, model revision, token limit, or hidden size changes.

The training objectives are:

1. policy-task binary cross-entropy;
2. workflow-family classification;
3. safe/attention counterfactual margin ranking; and
4. fact-preserving paraphrase consistency.

The Laya encoder remains frozen. The trainable artifact consists of three small heads rather than a
second copy of the base model.

### Local training environment

The included v2 model was trained and calibrated locally on a MacBook Air with an Apple M5 chip and
24 GB of unified memory, using PyTorch's MPS backend. No hosted GPU service was required. Training
is resumable: encoder features are checkpointed in small chunks, and each completed ensemble head
is saved separately.

## Evaluate

```bash
uv run legal-decision-model policy evaluate test challenge \
  --data benchmark \
  --model-dir policy-model
```

Reports are written to `evaluation/policy-v2/`. The suite measures:

- false clearances and attention recall;
- observed and upper-bounded false-clearance rates;
- clearance coverage and errors among cleared requests;
- counterfactual pair consistency;
- paraphrase invariance;
- safe-score monotonicity;
- out-of-distribution attention rate;
- workflow-family and generator results;
- behavioral-capability results; and
- routing reasons.

Accuracy is reported but is not the primary measure.

### Measured v2 results

| Split | Records | False clears | Attention recall | Clearance coverage | Pair consistency | False-clear upper bound |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 12,000 | 0 | 100% | 13.43% | Not measured | 0.0998% |
| Test | 10,000 | 0 | 100% | 13.43% | 26.86% | 0.1198% |
| Challenge | 5,000 | 0 | 100% | 5.16% | 10.32% | 0.2394% |

Validation met the policy's 0.1% upper-bound target using 3,000 independent attention-required
scenarios. Test and challenge observed no false clearances, but their 2,500 and 1,250 independent
attention-required scenarios yield upper bounds of 0.1198% and 0.2394%; neither independently
certifies the 0.1% target. Low pair consistency primarily reflects conservative escalation of safe
examples rather than missed risky examples. Full results, including paraphrase, monotonicity, OOD,
family, and capability metrics, are committed under `evaluation/policy-v2/`.

## How clearance works

The model predicts every policy prerequisite and escalation trigger. The clearance score is the
weakest required condition:

```text
minimum(
  prerequisite probabilities,
  1 - escalation-trigger probabilities,
  in-scope probability,
  1 - final-attention probability
)
```

The request is still routed to human attention when:

- any document window contains a sufficiently risky signal;
- the input is too long to process within the configured window limit;
- the embedding is too far from the validation distributions;
- the three heads disagree beyond the validation limit;
- global or workflow-family calibration lacks enough evidence; or
- the clearance score does not meet both applicable thresholds.

Threshold selection uses an exact one-sided Clopper-Pearson binomial upper confidence bound.
Correlated paraphrases from the same scenario count as one statistical trial; record-level coverage
remains separate. The default policy requires the global false-clearance upper bound to remain at or
below 0.1% with 95% confidence and permits no observed validation false-clearance scenarios at the
selected threshold. This is a finite-sample statement about the synthetic validation population,
not a guarantee on new data.

## Policy authoring

`policies/northstar.yaml` is the shared contract for generation, training, inference, and
evaluation. It defines facts, reasons, statistical limits, model settings, and long-document
behavior.

```bash
uv run legal-decision-model policy validate --file policies/northstar.yaml
```

See [docs/POLICY_AUTHORING.md](docs/POLICY_AUTHORING.md) before changing the policy. A policy change
requires a new benchmark, model artifact, and calibration.

## Customize it for your organization

Northstar is deliberately fictional. A business-specific version should begin with the
organization's approved decision policy, not with the model or this project's example triggers.

1. Define the exact routing decision. State what the system may clear, what must always reach a
   human, who owns the policy, and which team can approve changes.
2. Replace the Northstar prerequisites and triggers with concrete, observable facts from approved
   playbooks. Avoid broad labels such as "high risk" that different reviewers may interpret
   differently.
3. Update the workflow families, trigger language, safe examples, and behavioral transformations in
   the benchmark generator. Include the terminology, intake formats, jurisdictions, products, and
   escalation paths relevant to the organization without copying confidential material into a
   public project.
4. Generate synthetic counterfactuals first. For each safe request, change one material fact to
   create an attention-required partner, then add paraphrases, missing facts, misleading headings,
   long documents, conflicting accounts, and out-of-scope requests.
5. Keep final evaluation sources separate from development. Once a test or challenge set has
   influenced a design change, retire it as final evidence and evaluate against a newly generated
   or independently reviewed set.
6. Set the risk budget with the accountable legal owner. Retrain and recalibrate rather than
   copying Northstar's thresholds. Review global results and each important workflow, geography,
   business unit, language, or other approved evaluation group.
7. Introduce authorized organizational data only under separate privacy, security, privilege,
   retention, access, and governance controls. Use lawyer-reviewed examples and document how
   disagreements are resolved.
8. Run in shadow mode before changing any real route. Compare model proposals with human decisions,
   investigate every false clearance, preserve ordinary access to lawyers, and define override and
   rollback procedures.
9. Version the policy, benchmark, model, and thresholds together. Re-evaluate after policy changes,
   new products, new jurisdictions, intake-form changes, or shifts in request patterns.

The most useful organizational extension may be the policy and regression-testing system even if
automatic clearance is never enabled. See [Policy authoring](docs/POLICY_AUTHORING.md) and
[Practical in-house use](docs/IN_HOUSE_USE.md) for the detailed workflow and required controls.

## Architecture and report

- [Architecture](docs/ARCHITECTURE.md)
- [Practical in-house use](docs/IN_HOUSE_USE.md)
- [Technical report](TECHNICAL_REPORT.md)
- [Model card](MODEL_CARD.md)
- [Dataset card](DATASET_CARD.md)
- [Fictional contribution process](contributions/README.md)

## Preserved v0.1 baseline

The original 24-epoch, 10,000-record experiment is preserved for reproducibility:

| Split | Records | False clears | Attention recall | Clearance coverage | Pair consistency |
|---|---:|---:|---:|---:|---:|
| Validation | 1,000 | 0 | 100% | 24.2% | Not measured |
| Test | 1,000 | 0 | 100% | 31.5% | 63.0% |
| Challenge | 500 | 0 | 100% | 12.6% | 25.2% |

These are observed results on generated data. They do not establish a zero error rate or expected
performance on real requests.

## API

Start the server:

```bash
uv run legal-decision-model serve
```

Submit a fictional request:

```bash
curl -s http://127.0.0.1:8000/v1/decision \
  -H 'content-type: application/json' \
  -d '{"text":"The team cannot confirm what personal data leaves the device."}'
```

The API is a local demonstration. It has no authentication, authorization, rate limiting, audit
logging, tenant isolation, or production deployment configuration.

## Limits

- No real legal, customer, employee, contract, privileged, or internal company data was used.
- The fictional Northstar policy does not represent any company's actual legal policy.
- Synthetic benchmark performance may reflect generator regularities.
- Model task probabilities and reasons are not legal conclusions.
- The system cannot verify that a requester supplied complete or truthful facts.
- Validation-set risk bounds can fail under population or policy change.
- The project has not undergone the governance, privacy, security, or legal review required for
  operational use.
- Do not submit real or confidential material through the contribution process or local demo.

## Development

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run legal-decision-model validate --data data
uv run legal-decision-model scan-public
```

CI runs checks that do not download the Laya checkpoint. Full benchmark generation, training, and
evaluation remain explicit hardware-intensive commands.

## License and attribution

Project code and generated synthetic data are available under Apache License 2.0. The trained heads
are derived from the Apache-2.0 Laya checkpoint. See [NOTICE](NOTICE) for pinned package versions,
reviewed source revisions, and attribution.
