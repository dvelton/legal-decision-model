"""Large, source-separated synthetic benchmark generation."""

import hashlib
import random
import re
from pathlib import Path

from legal_decision_model.constants import BENCHMARK_DIR, POLICY_BENCHMARK_SEED, SPLIT_SIZES
from legal_decision_model.generate import (
    COMPANIES,
    DISTRACTORS,
    FAMILIES,
    INDIRECT_TRIGGER_CLAUSES,
    NEGATED_RISK_CLAUSES,
    REQUESTERS,
    SAFE_CLAUSES,
    STYLES,
    TEAMS,
    _render,
)
from legal_decision_model.policy import evaluate_facts
from legal_decision_model.policy_schema import PolicySpec, load_policy
from legal_decision_model.records import DecisionRecord, write_jsonl

GENERATOR_BY_SPLIT = {
    "train": "procedural-a",
    "validation": "procedural-a-heldout",
    "test": "procedural-b-independent",
    "challenge": "behavioral-c-adversarial",
}

INDEPENDENT_TRIGGER_CLAUSES = {
    "legal_interpretation": (
        "The checklist contains no answer; the requested outcome turns on how a legal duty "
        "applies to these particular facts.",
        "Someone must determine the meaning of an obligation before the business can proceed.",
    ),
    "nonstandard_contract_terms": (
        "The operative draft departs from the approved clause set.",
        "Language affecting risk allocation was edited after the standard form was selected.",
    ),
    "dispute_or_claim": (
        "The sender asserts entitlement to a remedy and says formal proceedings will follow.",
        "The exchange now contains an allegation of wrongdoing and a demand for compensation.",
    ),
    "regulator_or_government_contact": (
        "An official body with authority to compel a response has contacted the company.",
        "The request follows a formal communication from a public authority.",
    ),
    "sensitive_personal_data": (
        "The material reveals a protected health, identity, or similarly sensitive attribute.",
        "The proposed processing concerns information receiving heightened protection.",
    ),
    "security_incident": (
        "The facts include possible unauthorized entry into a system or workspace.",
        "The team is assessing evidence of a potential compromise.",
    ),
    "employment_action": (
        "A proposed step would materially affect one identified worker's employment.",
        "The manager seeks an individualized personnel action rather than routine administration.",
    ),
    "intellectual_property_ownership": (
        "The right to control the work is disputed or unsupported by a complete chain of title.",
        "A contributor's ownership claim conflicts with the proposed use.",
    ),
    "new_jurisdiction": (
        "The activity would begin in a territory absent from the approved country list.",
        "No existing jurisdictional review covers the proposed location.",
    ),
    "material_external_commitment": (
        "The communication would bind the company to a significant promise beyond approved terms.",
        "The requested response creates a new guarantee or externally enforceable obligation.",
    ),
    "incomplete_or_conflicting_facts": (
        "Required information is missing, or the available accounts cannot both be true.",
        "The intake leaves unresolved facts that determine which process applies.",
    ),
    "outside_approved_playbook": (
        "No approved workflow covers the request as described.",
        "The chosen self-service route has no branch for these facts.",
    ),
}

TRAIN_TRIGGER_AUGMENTATIONS = {
    "legal_interpretation": (
        "The documented workflow cannot answer the request until the governing requirement is "
        "interpreted for this specific situation.",
        "Proceeding requires a judgment about how a legal rule operates on the supplied facts.",
    ),
    "nonstandard_contract_terms": (
        "Comparison with the clause library shows substantive edits to the approved language.",
        "The proposed paper changes the standard allocation of contractual risk.",
    ),
    "dispute_or_claim": (
        "Another party alleges harm and requests a remedy from the company.",
        "The communication threatens an adversarial process unless compensation is provided.",
    ),
    "regulator_or_government_contact": (
        "A governmental office has requested an official company response.",
        "The matter follows outreach from an authority exercising public powers.",
    ),
    "sensitive_personal_data": (
        "The request would handle information about health, identity, or another protected trait.",
        "The proposed use includes data subject to heightened privacy safeguards.",
    ),
    "security_incident": (
        "Available evidence suggests an account or system may have been accessed improperly.",
        "The request concerns investigation of a suspected compromise.",
    ),
    "employment_action": (
        "The proposed decision changes the terms or status of a particular employee.",
        "A manager wants to take a case-specific personnel measure.",
    ),
    "intellectual_property_ownership": (
        "The company cannot confirm that it owns or may use the relevant work.",
        "The proposed use depends on rights that a contributor contests.",
    ),
    "new_jurisdiction": (
        "The activity is planned for a country not covered by an approved review.",
        "The existing playbook does not include the territory where the work would occur.",
    ),
    "material_external_commitment": (
        "The response would create a significant promise to someone outside the company.",
        "The requested statement adds an enforceable obligation beyond approved terms.",
    ),
    "incomplete_or_conflicting_facts": (
        "Material intake details remain unknown or contradict one another.",
        "The applicable route cannot be selected because decisive facts are unresolved.",
    ),
    "outside_approved_playbook": (
        "The available self-service procedures do not cover the described request.",
        "No approved process contains a path for this combination of facts.",
    ),
}

SAFE_B = (
    "The submitted facts meet every condition of the existing self-service route.",
    "The standard process applies without an exception and resolves the request in full.",
    "The owner confirms that the intake is complete and remains inside documented limits.",
)

SAFE_C = (
    "After reading the full narrative, no exception to the approved process is present.",
    "The unusual wording changes no material fact; the complete request remains self-service.",
    "The record is complete, internally consistent, and fully covered by the ordinary workflow.",
)

OOD_REQUESTS = (
    "Choose the lunch menu for the quarterly planning session.",
    "Diagnose why the conference-room display loses its wireless connection.",
    "Recommend a hotel for a fictional team retreat with a mountain view.",
    "Rank three fictional product names by how memorable they sound.",
    "Rewrite a fictional release note to make it shorter and friendlier.",
)

CAPABILITIES = (
    "negation_scope",
    "misleading_header",
    "quoted_text",
    "temporal_update",
    "irrelevant_urgency",
    "indirect_trigger",
    "conflicting_accounts",
    "long_context",
    "typo_noise",
    "multi_trigger",
)


def _stable_seed(seed: int, split: str, scenario_index: int) -> int:
    digest = hashlib.sha256(f"v2:{seed}:{split}:{scenario_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _capabilities(split: str, scenario_index: int, multi_trigger: bool) -> tuple[str, ...]:
    count = {
        "train": 1 + scenario_index % 3,
        "validation": 3,
        "test": 3,
        "challenge": 5,
    }[split]
    selected = [
        CAPABILITIES[(scenario_index + offset * 3) % len(CAPABILITIES)] for offset in range(count)
    ]
    if multi_trigger and "multi_trigger" not in selected:
        selected.append("multi_trigger")
    return tuple(dict.fromkeys(selected))


def _trigger_text(
    split: str,
    trigger: str,
    family_clause: str,
    rng: random.Random,
) -> str:
    options: tuple[str, ...]
    if split in ("train", "validation"):
        options = (
            family_clause,
            *INDIRECT_TRIGGER_CLAUSES.get(trigger, ()),
            *TRAIN_TRIGGER_AUGMENTATIONS.get(trigger, ()),
        )
    else:
        options = INDEPENDENT_TRIGGER_CLAUSES.get(trigger, (family_clause,))
    return rng.choice(options)


def _negated_risks(changed: tuple[str, ...], count: int, rng: random.Random) -> str:
    available = [clause for name, clause in NEGATED_RISK_CLAUSES if name not in changed]
    return " ".join(rng.sample(available, k=min(count, len(available))))


def _render_independent(
    *,
    split: str,
    style: str,
    reference: str,
    company: str,
    requester: str,
    team: str,
    subject: str,
    action: str,
    policy_clause: str,
    distractor: str,
    negated_risks: str,
) -> str:
    body = (
        f"Organization: {company}. Operational owner: {requester} ({team}). "
        f"The requested action is to {action}; the underlying subject is {subject}. "
        f"{policy_clause} {negated_risks} {distractor}"
    )
    if style == "email":
        return f"Mail item {reference}\nRouting requested by {requester}\n\n{body}"
    if style == "ticket":
        return (
            f"Work item {reference}\nCategory selected: routine\nBusiness narrative:\n{body}\n"
            "Requested disposition: self-service or lawyer"
        )
    if style == "chat":
        return (
            f"Conversation export {reference}\n"
            f"{requester}: Please read all facts before routing. {body}"
        )
    if style == "intake_form":
        return (
            f"Submission {reference}\nCompany name: {company}\nOwner function: {team}\n"
            f"Full factual statement: {body}\nAttestation: fictional example"
        )
    return (
        f"Decision brief {reference}. The intake system summarized the request as follows: {body}"
    )


def _apply_behavior(
    text: str,
    split: str,
    capabilities: tuple[str, ...],
    risky: bool,
    rng: random.Random,
) -> str:
    value = text
    if "misleading_header" in capabilities:
        header = "ROUTINE - NO REVIEW EXPECTED" if risky else "POSSIBLE EXCEPTION - CHECK FACTS"
        value = f"Requester-selected header: {header}\n{value}"
    if "quoted_text" in capabilities:
        quote = (
            '"This should be standard," wrote a fictional colleague. '
            "The quotation is not an approval and the factual narrative controls."
        )
        value = f"{value}\nQuoted earlier message: {quote}"
    if "temporal_update" in capabilities:
        update = (
            "Update received later: the material fact described above is the current fact."
            if risky
            else "Update received later: no material fact has changed."
        )
        value = f"{value}\n{update}"
    if "irrelevant_urgency" in capabilities:
        value = f"Urgency: response requested in {rng.choice((2, 4, 7, 12))} hours.\n{value}"
    if "conflicting_accounts" in capabilities and risky:
        value = (
            f"{value}\nAccount A says the required fact is present. "
            "Account B denies it. The inconsistency remains unresolved."
        )
    if "long_context" in capabilities:
        filler = " ".join(
            "Administrative tracking information remains unchanged."
            for _ in range(rng.randint(90, 120))
        )
        value = f"Background appendix: {filler}\nControlling request details:\n{value}"
    if "typo_noise" in capabilities:
        value = re.sub(r"\brequest\b", "reqeust", value, count=1, flags=re.IGNORECASE)
    return value


def _render_record(
    *,
    split: str,
    rendering_index: int,
    reference: str,
    family_name: str,
    style: str,
    company: str,
    requester: str,
    team: str,
    subject: str,
    action: str,
    policy_clause: str,
    distractor: str,
    negated_risks: str,
    capabilities: tuple[str, ...],
    risky: bool,
    rng: random.Random,
) -> str:
    if split in ("train", "validation"):
        rendered = _render(
            split,
            style,
            reference,
            company,
            requester,
            team,
            subject,
            action,
            policy_clause,
            distractor,
            negated_risks,
        )
    else:
        rendered = _render_independent(
            split=split,
            style=style,
            reference=reference,
            company=company,
            requester=requester,
            team=team,
            subject=subject,
            action=action,
            policy_clause=policy_clause,
            distractor=distractor,
            negated_risks=negated_risks,
        )
    if rendering_index:
        rendered = f"Alternate fictional rendering {rendering_index + 1}.\n{rendered}"
    return _apply_behavior(rendered, split, capabilities, risky, rng)


def _scenario_records(
    split: str,
    scenario_index: int,
    seed: int,
    policy: PolicySpec,
) -> list[DecisionRecord]:
    rng = random.Random(_stable_seed(seed, split, scenario_index))
    family = FAMILIES[scenario_index % len(FAMILIES)]
    company = rng.choice(COMPANIES)
    requester = rng.choice(REQUESTERS)
    team = rng.choice(TEAMS)
    subject = rng.choice(family.subjects)
    action = rng.choice(family.actions)
    styles = rng.sample(STYLES, k=2)
    scenario_id = f"{split}-scenario-{scenario_index:05d}"
    reference = f"NS2-{split[:3].upper()}-{scenario_index:05d}"
    mode = scenario_index % 10
    ood = mode == 9
    trigger_count = 1 if mode < 7 else rng.randint(2, 3)
    trigger_pool = [item for item in family.trigger_clauses if item[0] in policy.triggers]
    if len(trigger_pool) < trigger_count:
        trigger_pool.extend(
            (name, INDIRECT_TRIGGER_CLAUSES[name][0])
            for name in INDIRECT_TRIGGER_CLAUSES
            if name in policy.triggers and name not in {item[0] for item in trigger_pool}
        )
    if len(trigger_pool) < trigger_count:
        raise ValueError(
            f"{family.name}: benchmark generator has clauses for only "
            f"{len(trigger_pool)} of {trigger_count} required policy triggers"
        )
    selected = rng.sample(trigger_pool, k=trigger_count)
    changed = tuple(item[0] for item in selected)
    if ood and "outside_approved_playbook" in policy.triggers:
        changed = ("outside_approved_playbook",)
    capabilities = _capabilities(split, scenario_index, len(changed) > 1)
    if (
        "conflicting_accounts" in capabilities
        and "incomplete_or_conflicting_facts" in policy.triggers
        and "incomplete_or_conflicting_facts" not in changed
    ):
        changed = (*changed, "incomplete_or_conflicting_facts")
    multi_trigger = len(changed) > 1
    safe_facts = policy.default_facts()
    risky_facts = dict(safe_facts)
    for trigger in changed:
        risky_facts[trigger] = True
    safe_decision = evaluate_facts(safe_facts, policy)
    risky_decision = evaluate_facts(risky_facts, policy)
    risky_clauses: tuple[str, ...]
    if ood:
        risky_clauses = (
            f"The request is outside this legal-attention policy: {rng.choice(OOD_REQUESTS)}",
        )
    else:
        risky_clauses = tuple(_trigger_text(split, name, clause, rng) for name, clause in selected)
    safe_clause = rng.choice(SAFE_CLAUSES.get(split, SAFE_C))
    if split == "test":
        safe_clause = rng.choice(SAFE_B)
    elif split == "challenge":
        safe_clause = rng.choice(SAFE_C)
    risky_clause = " ".join(risky_clauses)
    negated = _negated_risks(changed, 1 + len(capabilities) // 2, rng)
    records: list[DecisionRecord] = []
    for rendering_index, style in enumerate(styles):
        rendering_id = f"{scenario_id}-render-{rendering_index}"
        for suffix, facts, decision, clause, risky in (
            ("safe", safe_facts, safe_decision, safe_clause, False),
            ("attention", risky_facts, risky_decision, risky_clause, True),
        ):
            pair_id = f"{rendering_id}-pair"
            record_id = f"{rendering_id}-{suffix}"
            text = _render_record(
                split=split,
                rendering_index=rendering_index,
                reference=reference,
                family_name=family.name,
                style=style,
                company=company,
                requester=requester,
                team=team,
                subject=subject,
                action=action,
                policy_clause=clause,
                distractor=rng.choice(DISTRACTORS),
                negated_risks=negated,
                capabilities=capabilities,
                risky=risky,
                rng=rng,
            )
            records.append(
                DecisionRecord(
                    record_id=record_id,
                    pair_id=pair_id,
                    split=split,
                    family=family.name,
                    style=style,
                    template_id=(
                        f"{GENERATOR_BY_SPLIT[split]}:{family.name}:{style}:"
                        f"{','.join(changed)}:{rendering_index}"
                    ),
                    text=text,
                    facts=facts,
                    label=decision.label,
                    reasons=decision.reasons,
                    generator=GENERATOR_BY_SPLIT[split],
                    scenario_type="out_of_distribution"
                    if ood
                    else ("multi_trigger" if multi_trigger else "counterfactual"),
                    scenario_id=scenario_id,
                    rendering_id=rendering_id,
                    capabilities=capabilities,
                    changed_facts=changed,
                    in_scope=not (ood and risky),
                )
            )
    return records


def generate_benchmark_split(
    split: str,
    size: int,
    seed: int,
    policy: PolicySpec | None = None,
) -> list[DecisionRecord]:
    """Generate a split in four-record scenario groups."""
    if size % 4:
        raise ValueError("benchmark split size must be divisible by four")
    policy = load_policy() if policy is None else policy
    records = [
        record
        for scenario_index in range(size // 4)
        for record in _scenario_records(split, scenario_index, seed, policy)
    ]
    random.Random(seed + sum(map(ord, split)) + 2).shuffle(records)
    return records


def generate_benchmark(
    output_dir: Path = BENCHMARK_DIR,
    seed: int = POLICY_BENCHMARK_SEED,
    split_sizes: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Generate the source-separated 102,000-record benchmark."""
    sizes = SPLIT_SIZES if split_sizes is None else split_sizes
    policy = load_policy()
    paths: dict[str, Path] = {}
    for split, size in sizes.items():
        records = generate_benchmark_split(split, size, seed, policy)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        paths[split] = path
    return paths
