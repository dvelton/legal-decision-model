# Legal Decision Model

Legal Decision Model is a local proof of concept with one job: read a fictional corporate request
and decide whether it requires human lawyer attention.

The project generates its own synthetic data, trains a small decision head on the Apache-licensed
[Laya](https://huggingface.co/convaiinnovations/laya) encoder, and applies a conservative clearance
threshold. A request receives `NO_HUMAN_LAWYER_ATTENTION` only when the model is sufficiently
confident. Everything else receives `REQUIRES_HUMAN_LAWYER_ATTENTION`.

This is a demonstration, not legal advice. It has not been tested on real legal matters and should
not be used to bypass legal, compliance, privacy, security, employment, or regulatory review.

## What you can do with it

1. Generate 10,000 fictional requests and deterministic labels.
2. Inspect the fictional Northstar Legal Attention Policy that creates those labels.
3. Train a task-specific decision head on a Mac, CPU, or CUDA device.
4. Measure false clearances, attention recall, calibration, clearance coverage, and
   counterfactual-pair consistency.
5. Try the trained model through a command line or a local browser interface.

## Quick start

Requirements:

- macOS, Linux, or Windows
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- About 5 GB of free disk space for dependencies, the Laya checkpoint, and local artifacts

```bash
git clone https://github.com/dvelton/legal-decision-model.git
cd legal-decision-model
uv sync
uv run legal-decision-model validate
uv run legal-decision-model evaluate
uv run legal-decision-model serve
```

Open `http://127.0.0.1:8000`. The repository includes the trained task head and metadata; Laya's
base checkpoint downloads on first use.

To regenerate the data and retrain:

```bash
uv run legal-decision-model generate
uv run legal-decision-model train --device mps
uv run legal-decision-model evaluate
```

Use `--device cpu` on a machine without Apple Silicon or CUDA. The first training run downloads the base checkpoint. Training time depends on hardware.

## Make one decision

```bash
uv run legal-decision-model decide --text \
  "The counterparty inserted a new unlimited liability clause into the standard order form."
```

The output includes both class probabilities, the safe-clearance threshold, latency, the base
model name, policy version, and routing reason.

## How it works

```text
structured fictional facts
  -> deterministic fictional policy label
  -> varied fictional email, ticket, chat, form, or summary
  -> frozen Laya encoder
  -> trained stuntd decision head
  -> validation-set temperature calibration
  -> conservative safe-clearance threshold
  -> binary routing decision
```

The label does not come from an LLM. `policy.py` applies explicit fictional rules first. The text
renderer then turns those labeled facts into a request. This keeps the source of truth inspectable
and avoids using real legal data.

Each safe request has a counterfactual partner in which one policy fact changes. For example, an
approved template request may become a lawyer-attention request solely because a nonstandard
liability clause was added.

## Fictional policy

The synthetic policy routes a request to human lawyer attention when it involves any of these
conditions:

- individualized legal interpretation
- nonstandard contract terms
- a dispute, claim, regulator, or government authority
- sensitive personal data or a security incident
- an individualized employment action
- intellectual-property ownership
- a jurisdiction outside the approved playbook
- a material external commitment
- incomplete or conflicting facts
- no self-service process that fully resolves the request

A request can receive `NO_HUMAN_LAWYER_ATTENTION` only when an approved self-service process fully
resolves a complete request and none of the escalation conditions apply.

## Dataset

| Split | Records | Purpose |
|---|---:|---|
| Train | 7,500 | Fit the task-specific head |
| Validation | 1,000 | Calibrate probabilities and select the clearance threshold |
| Test | 1,000 | Measure held-out performance |
| Challenge | 500 | Stress negation, indirect descriptions, and unfamiliar phrasing |

The data covers contracts, privacy, security, intellectual property, marketing, employment,
disputes, corporate work, product review, and legal operations. All organizations, people,
products, amounts, facts, and documents are fictional.

See [DATASET_CARD.md](DATASET_CARD.md) for the generation and split methodology.

## Evaluation standard

Ordinary accuracy is reported but is not the primary measure. The project gives priority to:

- false-clear rate: matters requiring attention that the model clears
- attention recall: share of attention-required matters routed to a lawyer
- clearance error rate: errors among the requests that the model clears
- clearance coverage: share of all requests the model clears
- counterfactual-pair consistency
- calibration measured with the Brier score

The default threshold allows no false clearances on the synthetic validation split. That is a
threshold-selection rule, not a claim that future or real-world errors will be zero.

Calibration and threshold selection run against the serialized float16 head through the same
serving path used for inference. If no threshold can satisfy the configured false-clear limit,
clearance is disabled and every request is routed to human attention.

See [MODEL_CARD.md](MODEL_CARD.md), `evaluation/current/`, and the preserved
`evaluation/baseline-3-epoch/` run for measured results.

### Current measured results

| Split | Records | False clears | Attention recall | Clearance coverage | Pair consistency |
|---|---:|---:|---:|---:|---:|
| Validation | 1,000 | 0 | 100% | 24.2% | Not measured |
| Test | 1,000 | 0 | 100% | 31.5% | 63.0% |
| Challenge | 500 | 0 | 100% | 12.6% | 25.2% |

The challenge split uses more indirect language, unfamiliar formatting, and explicit negations of
unrelated risks. The model responds by escalating most challenge requests. Zero observed false
clearances on these generated splits does not establish a zero error rate on other data.

Compared with the preserved 3-epoch baseline, the expanded 24-epoch experiment increased test
clearance coverage by 3.25 percentage points and challenge coverage by 5.27 points while retaining
zero observed false clearances. Because both the data and epoch count changed, the comparison does
not isolate which change produced the gain.

## Commands

| Command | Action |
|---|---|
| `generate` | Rebuild all synthetic data from a fixed seed |
| `validate` | Check labels, pairs, duplicates, split isolation, and synthetic-only constraints |
| `train` | Train, calibrate, and save the task head |
| `evaluate` | Evaluate test and challenge splits |
| `decide` | Make one command-line decision |
| `serve` | Run the local browser interface and API |
| `scan-public` | Flag likely internal URLs, paths, credentials, and private references |

Run `uv run legal-decision-model <command> --help` for command options.

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

`routing_reason` is `MODEL_DECISION` for ordinary inference and
`INPUT_EXCEEDS_MODEL_TOKEN_BUDGET` when the complete request cannot fit. The latter always routes
to human attention.

The API is intended for local demonstration. It has no authentication, rate limiting, audit
logging, tenant isolation, or production deployment configuration.

## Limits

- The training data is synthetic and simplified.
- The fictional policy does not represent any company's actual legal policy.
- The model may learn wording patterns that do not transfer to real requests.
- A binary output hides the reason a real lawyer may need to review a matter.
- Inputs that exceed the model's actual token budget are routed to human attention before any
  truncated model decision can be made.
- The model does not identify law, give advice, interpret contracts, or replace professional
  judgment.
- A validation threshold cannot establish safety on new populations.
- The model should not process privileged, confidential, personal, customer, employee, or
  production data.

## Development

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run legal-decision-model scan-public
```

The continuous-integration workflow runs checks that do not download the model checkpoint. Model
training and evaluation are separate because they are hardware-intensive.

## License and attribution

Project code and synthetic data are available under Apache License 2.0. The trained head is derived
from the Apache-2.0 Laya checkpoint. See [NOTICE](NOTICE) for pinned upstream versions, source
revisions, and attribution.