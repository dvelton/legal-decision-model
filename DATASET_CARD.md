# Dataset Card: Northstar Legal Attention

## Summary

Northstar Legal Attention is a generated dataset for one binary decision: whether a fictional
corporate request requires human lawyer attention.

Every record is synthetic. Company names, people, teams, documents, events, jurisdictions, and
request facts are invented for this project.

## Source of labels

Labels come from the executable policy in `src/legal_decision_model/policy.py`. The policy evaluates
structured boolean facts. It does not ask a language model to choose the label.

The text generator runs only after the label is known:

```text
structured facts -> deterministic policy -> label -> fictional prose
```

This separates policy correctness from prose generation.

## Schema

Each JSONL record contains:

| Field | Meaning |
|---|---|
| `record_id` | Unique record identifier |
| `pair_id` | Counterfactual-pair identifier |
| `split` | Train, validation, test, or challenge |
| `family` | Fictional workflow family |
| `style` | Email, ticket, chat, intake form, or summary |
| `template_id` | Split-isolated renderer identifier |
| `text` | Model input |
| `facts` | Structured fictional policy facts |
| `label` | Deterministic binary decision |
| `reasons` | Policy reasons used for validation, not model input |
| `synthetic` | Always `true` |

## Size

| Split | Records | Counterfactual pairs |
|---|---:|---:|
| Train | 7,500 | 3,750 |
| Validation | 1,000 | 500 |
| Test | 1,000 | 500 |
| Challenge | 500 | 250 |
| Total | 10,000 | 5,000 |

Each split is balanced between the two labels.

## Content

The dataset covers ten fictional workflow families:

1. contracts
2. privacy
3. security
4. intellectual property
5. marketing
6. employment
7. disputes
8. corporate work
9. product review
10. legal operations

The generated requests use five presentation styles, direct and indirect descriptions of each
escalation condition, neutral details, and explicit negations of unrelated risk conditions. Test
and challenge records contain more negated-risk distractors than training records.

## Counterfactual construction

Each pair shares the same fictional company, requester, team, subject, action, style, and distractor.
The pair differs in exactly one policy fact. One record stays within an approved self-service
process. The other adds a condition such as a dispute, sensitive data, a nonstandard term, or
incomplete information.

This design tests whether the model responds to the legally relevant fictional fact rather than
unrelated wording.

## Split methodology

The generator assigns renderer identifiers within a split and validates that no `template_id`
appears in more than one split. Records are shuffled deterministically with a fixed seed. Normalized
request text and record IDs must be unique across the full dataset.

The validation split is reserved for calibration and threshold selection. Test and challenge labels
are not used during training or threshold selection.

## Validation

`legal-decision-model validate` checks:

- exact split sizes
- deterministic policy-label agreement
- policy-reason agreement
- unique IDs and normalized texts
- split-isolated template identifiers
- two records and both labels in every pair
- exactly one changed policy fact per pair
- absence of selected internal URLs, corporate email addresses, credentials, and local paths

## Known limitations

The data is artificial and does not reproduce real legal distributions, drafting styles, factual
uncertainty, adversarial behavior, or policy complexity. Balanced labels are useful for experiments
but are unlikely to reflect a real intake queue. The text generator may leave detectable patterns
that make the task easier than real routing.

This dataset must not be represented as anonymized, de-identified, or representative legal data. It
contains no source legal matters to anonymize.
