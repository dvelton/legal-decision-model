# Policy authoring

The model learns a fictional routing policy rather than an open-ended definition of legal work.
The policy file is therefore the main product specification.

## File structure

Create or edit a YAML file with these sections:

```yaml
name: Example Legal Attention Policy
version: example-1.0
description: Fictional policy used for a synthetic experiment.

labels:
  requires_attention: REQUIRES_HUMAN_LAWYER_ATTENTION
  no_attention: NO_HUMAN_LAWYER_ATTENTION

prerequisites:
  approved_self_service_process:
    default: true
    reason: No approved self-service process fully resolves the request.

triggers:
  nonstandard_contract_terms:
    reason: The request includes nonstandard contract terms.

clearance:
  success_reason: An approved self-service process fully resolves the complete request.
  require_all_prerequisites: true
  require_no_triggers: true
```

The committed Northstar policy also includes `risk_control` and `model` sections. See
`policies/northstar.yaml` for the complete schema.

## Design rules

Use facts that can be stated as booleans and that have a clear operational meaning. A trigger such
as `security_incident` is testable. A trigger such as `high risk` is too vague because labelers and
requesters may interpret it differently.

Keep prerequisites separate from escalation triggers:

- a prerequisite must be true before self-service clearance is possible;
- a trigger routes the request to human attention when it is true.

Reasons should explain the routing fact in plain language. They are returned for inspection but are
not supplied as model input.

## Versioning

Change the policy version whenever a fact, reason, threshold, or operating rule changes. A trained
artifact stores a fingerprint of the complete compiled policy, including model and statistical
settings. The loader rejects any mismatch before serving.

## Validation

Run:

```bash
uv run legal-decision-model policy validate --file policies/northstar.yaml
```

Validation rejects missing sections, unknown value types, invalid probability bounds, non-positive
model dimensions, and window overlap that consumes the full token budget.

## Regeneration and retraining

A policy change requires a new benchmark and model artifact:

```bash
uv run legal-decision-model policy generate --output benchmark
uv run legal-decision-model policy validate-benchmark --data benchmark
uv run legal-decision-model policy train --data benchmark --device mps
uv run legal-decision-model policy evaluate test challenge --data benchmark
```

Do not reuse thresholds or heads from a different policy version.

## Adapting the design for an internal proof of concept

An internal team could replace Northstar with an approved fictional or abstracted policy and keep
the same pipeline. Before using representative or operational data, the team would need separate
authorization for data handling, privacy, security, privilege, retention, access, governance, and
human oversight. This repository does not supply those approvals.
