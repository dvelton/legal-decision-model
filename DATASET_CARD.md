# Dataset Card: Northstar Legal Attention Benchmark

## Summary

The Northstar Legal Attention Benchmark is a reproducible, 102,000-record synthetic benchmark for
one binary decision: whether a fictional corporate request requires human lawyer attention.

Every organization, person, document, event, jurisdiction, and request fact is invented for this
project. The benchmark does not contain anonymized or de-identified legal data because no source
legal matters were used.

## Source of labels

`policies/northstar.yaml` defines explicit boolean prerequisites and escalation triggers.
`policy.py` applies those rules to structured fictional facts before prose is generated:

```text
structured fictional facts
  -> deterministic fictional policy
  -> binary label and reasons
  -> fictional prose rendering
```

No language model chooses or verifies the benchmark label.

## Size

| Split | Records | Scenarios | Counterfactual pairs | Generator |
|---|---:|---:|---:|---|
| Train | 75,000 | 18,750 | 37,500 | Procedural A |
| Validation | 12,000 | 3,000 | 6,000 | Procedural A, held out |
| Test | 10,000 | 2,500 | 5,000 | Independent procedural B |
| Challenge | 5,000 | 1,250 | 2,500 | Behavioral adversarial C |
| Total | 102,000 | 25,500 | 51,000 | Three source families |

The two labels and ten workflow families are balanced across the full benchmark.

## Four-record scenario design

Each scenario contains:

1. a safe request in rendering A;
2. an attention-required counterfactual in rendering A;
3. the same safe facts in rendering B; and
4. the same attention-required facts in rendering B.

This structure supports counterfactual testing and paraphrase testing without changing the source
facts.

## Scenario types

| Type | Records | Purpose |
|---|---:|---|
| Single-trigger counterfactual | 47,800 | Isolate one policy-relevant change |
| Multi-trigger | 44,000 | Test multiple simultaneous escalation facts |
| Out of distribution | 10,200 | Test requests outside the policy's intended scope |

The out-of-distribution examples include fictional non-legal requests such as choosing a lunch
menu, diagnosing a conference-room display, or recommending a hotel.

## Behavioral capabilities

The benchmark contains 24,100 tagged records for each of these capabilities:

- negation scope;
- misleading requester-selected headings;
- quoted text that does not control the current facts;
- later factual updates;
- irrelevant urgency;
- indirect trigger descriptions;
- conflicting accounts;
- long context;
- typo noise; and
- multi-trigger combinations.

A record can carry more than one capability tag.

## Source separation

Training and validation use generator A. The test split uses independently authored clauses and a
different document structure in generator B. The challenge split adds behavioral transformations
through generator C.

The validator rejects reuse of a training generator in test or challenge. It also rejects a
`template_id` appearing in more than one split.

Source separation reduces direct renderer leakage but does not make the synthetic benchmark
representative of real legal intake.

## Schema

Each JSONL record contains:

| Field | Meaning |
|---|---|
| `record_id` | Unique record identifier |
| `pair_id` | Safe/attention counterfactual-pair identifier |
| `scenario_id` | Four-record scenario identifier |
| `rendering_id` | Fact-preserving rendering identifier |
| `split` | Train, validation, test, or challenge |
| `generator` | Source generator for split-separation checks |
| `scenario_type` | Counterfactual, multi-trigger, or out of distribution |
| `family` | Fictional workflow family |
| `style` | Email, ticket, chat, intake form, or summary |
| `template_id` | Split-isolated renderer identifier |
| `text` | Model input |
| `facts` | Structured fictional policy facts |
| `label` | Deterministic binary decision |
| `reasons` | Deterministic policy reasons, excluded from model input |
| `capabilities` | Behavioral test tags |
| `changed_facts` | Policy facts changed within the counterfactual pair |
| `in_scope` | Whether the request is within the fictional policy scope |
| `synthetic` | Always `true` |

## Reproduction

The full benchmark is generated locally because the JSONL files are large:

```bash
uv run legal-decision-model policy generate --output benchmark --seed 20261004
uv run legal-decision-model policy validate-benchmark --data benchmark
```

Generation is deterministic for the committed code, policy, and seed. The expected counts are
recorded in `evaluation/policy-v2/benchmark-manifest.json`.

## Validation

The validator checks:

- exact split sizes;
- deterministic policy-label and reason agreement;
- unique IDs and normalized text;
- public-safe synthetic content;
- source-generator separation;
- split-isolated templates;
- two records and both labels in every pair;
- exactly four records and two renderings in every scenario;
- fact preservation across paraphrases;
- changed-fact metadata;
- out-of-scope fact consistency; and
- declared capability tags.

## Fictional human-review contributions

`contributions/` defines a public contribution format for newly written fictional behavioral cases.
A submitted case is not counted as human-reviewed until a reviewer records an accepted status,
reviewer name, date, and policy version. No contributed case is included in the generated benchmark
unless it is explicitly incorporated into generation code.

## Limitations

- Synthetic language and balanced labels do not reproduce an operational intake queue.
- Deterministic facts cannot model whether a requester omitted, misunderstood, or misstated facts.
- Source-separated generators still share project vocabulary and policy structure.
- Behavioral tags test selected transformations rather than the full range of adversarial input.
- Benchmark performance cannot establish safety, legal adequacy, or production fitness.
- The benchmark must not be represented as real, anonymized, de-identified, or representative
  legal data.
