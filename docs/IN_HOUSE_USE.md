# Practical use for in-house legal teams

This project is not ready to route real matters. Its practical value is as a policy and evaluation
tool that an in-house team can test before deciding whether operational automation is appropriate.

## Useful proof-of-concept workflows

### 1. Shadow-mode intake routing

Run the model beside the existing intake process without changing where requests go. Compare the
model's proposed route with the actual human route and investigate disagreements. No requester or
lawyer should lose access to the ordinary process during shadow testing.

Questions the experiment can answer:

- Which request types have a sufficiently clear self-service boundary?
- Which wording or formats cause avoidable escalation?
- Which policy conditions are commonly missing from intake?
- How often do model members disagree or identify out-of-distribution input?

### 2. Self-service policy testing

Before training a model, the executable policy forces the team to state what must be true before a
request can be handled through self-service. Synthetic counterfactuals can test whether one changed
fact should change the route.

This can expose unclear policy boundaries even if the team never deploys the model.

### 3. Regression testing

Use the benchmark as a policy regression suite. When a policy, intake form, playbook, or model
changes, rerun:

- deterministic label checks;
- counterfactual pair consistency;
- paraphrase invariance;
- long-document behavior;
- OOD and disagreement gates; and
- statistical risk thresholds.

The result is a documented test of what changed rather than a general claim that the system
improved.

### 4. Intake-form design

Low prerequisite probabilities or recurring incomplete-fact triggers can identify questions that a
fictional intake form should ask more directly. In a real project, humans would confirm that the
signal reflects missing intake design rather than model error before changing the form.

### 5. Human queue support

A mature, separately governed system might use conservative routing signals to organize a human
queue, flag missing information, or identify requests that appear to fit an approved self-service
path. The human remains able to override the route, and the system must not present a clearance as
legal advice.

## Controls required before real data

An internal team would need to define and approve:

- the business owner and accountable legal owner;
- authorized data sources and prohibited data;
- privilege, confidentiality, privacy, security, retention, and access controls;
- the precise policy and its versioning process;
- representative evaluation populations;
- minimum evidence for global and subgroup clearance;
- human override and escalation paths;
- monitoring for population, policy, and performance drift;
- incident response and rollback;
- requester disclosures and interface language; and
- periodic human review of errors and over-escalations.

The model should begin in shadow mode. A later move to assisted or automated routing should require
new approval based on measured performance and operational controls.

## Strongest objections

### Synthetic performance may not transfer

This is the central limitation. Source-separated generators and behavioral tests reduce direct
template leakage but cannot establish performance on actual intake. Representative authorized data
and human review would be required for that question.

### A binary route hides legal complexity

The external answer is intentionally narrow. The internal fact outputs improve inspection but do
not replace a lawyer's assessment of law, strategy, privilege, materiality, or business context.

### Automation can create false confidence

The fail-closed thresholds, OOD gate, disagreement gate, and disclaimer reduce that risk but do not
remove it. Interface design, training, governance, audit, and continuing human access matter as
much as model accuracy.

## Appropriate success criteria

For an internal proof of concept, success would mean:

- the policy can be stated and versioned clearly;
- humans agree on the deterministic synthetic cases;
- held-out false-clearance bounds meet the approved target;
- over-escalation is low enough to provide practical value;
- counterfactuals and paraphrases behave consistently;
- unsafe and unfamiliar inputs fail closed; and
- the team can explain, reproduce, monitor, and disable the system.

Success would not mean that a model has learned law or that lawyers are no longer needed.
