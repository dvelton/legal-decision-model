"""Deterministic synthetic-data generation."""

import hashlib
import random
from dataclasses import dataclass, replace
from pathlib import Path

from legal_decision_model.constants import DATA_DIR, SPLIT_SIZES
from legal_decision_model.policy import ScenarioFacts, evaluate_policy
from legal_decision_model.records import DecisionRecord, write_jsonl


@dataclass(frozen=True)
class FamilySpec:
    """Text ingredients and relevant escalation triggers for one fictional workflow."""

    name: str
    subjects: tuple[str, ...]
    actions: tuple[str, ...]
    trigger_clauses: tuple[tuple[str, str], ...]


FAMILIES = (
    FamilySpec(
        "contracts",
        ("software subscription", "vendor order form", "consulting statement of work"),
        (
            "use the pre-approved template without edits",
            "renew on the existing approved commercial terms",
            "complete the standard signature-routing steps",
        ),
        (
            (
                "nonstandard_contract_terms",
                "The counterparty inserted a new unlimited liability clause.",
            ),
            (
                "material_external_commitment",
                "The draft promises a public launch date and service credit.",
            ),
            (
                "outside_approved_playbook",
                "The request uses a paper that is not in the approved catalog.",
            ),
            (
                "legal_interpretation",
                "The requester asks whether the indemnity language covers this loss.",
            ),
        ),
    ),
    FamilySpec(
        "privacy",
        ("product analytics setting", "customer profile field", "research participant form"),
        (
            "follow the approved data-minimization checklist",
            "use the documented retention setting",
            "apply the standard consent configuration",
        ),
        (
            (
                "sensitive_personal_data",
                "The proposed field records medical accommodation details.",
            ),
            (
                "new_jurisdiction",
                "The feature will be offered for the first time in a new country.",
            ),
            (
                "outside_approved_playbook",
                "The data use is not listed in the approved privacy playbook.",
            ),
            (
                "incomplete_or_conflicting_facts",
                "The team cannot confirm what data leaves the device.",
            ),
        ),
    ),
    FamilySpec(
        "security",
        ("access review", "security questionnaire", "authentication configuration"),
        (
            "use the approved security response library",
            "complete the standard access-review workflow",
            "apply the documented authentication baseline",
        ),
        (
            (
                "security_incident",
                "Logs suggest that an unknown party accessed a customer workspace.",
            ),
            (
                "regulator_or_government_contact",
                "A government cyber office sent a preservation request.",
            ),
            (
                "material_external_commitment",
                "The response would promise a security control not yet deployed.",
            ),
            (
                "incomplete_or_conflicting_facts",
                "The available logs conflict about whether data was accessed.",
            ),
        ),
    ),
    FamilySpec(
        "intellectual_property",
        ("open-source intake", "design asset request", "contractor deliverable"),
        (
            "use a component already listed in the approved catalog",
            "use artwork from the licensed internal library",
            "record the deliverable under the standard assignment process",
        ),
        (
            (
                "intellectual_property_ownership",
                "Two contributors claim ownership of the same deliverable.",
            ),
            (
                "outside_approved_playbook",
                "The proposed license is absent from the approved catalog.",
            ),
            ("dispute_or_claim", "A third party alleges that the asset copies its protected work."),
            (
                "legal_interpretation",
                "The team asks whether this license permits the planned distribution.",
            ),
        ),
    ),
    FamilySpec(
        "marketing",
        ("customer quote", "comparison webpage", "promotional giveaway"),
        (
            "publish the already approved wording unchanged",
            "use the approved factual comparison table",
            "run the standard low-value giveaway process",
        ),
        (
            (
                "material_external_commitment",
                "The copy guarantees a result that the product team cannot verify.",
            ),
            ("dispute_or_claim", "A competitor has demanded removal of the comparison."),
            ("new_jurisdiction", "The campaign will run in a country not covered by the playbook."),
            ("outside_approved_playbook", "The team wants to use a new endorsement format."),
        ),
    ),
    FamilySpec(
        "employment",
        ("routine leave request", "standard role change", "manager training request"),
        (
            "follow the published self-service instructions",
            "use the standard role-change checklist with no individual exception",
            "assign the approved manager course",
        ),
        (
            (
                "employment_action",
                "The manager proposes terminating one employee after a complaint.",
            ),
            ("sensitive_personal_data", "The request includes an employee's medical diagnosis."),
            ("dispute_or_claim", "The employee has threatened a formal claim."),
            (
                "incomplete_or_conflicting_facts",
                "The manager and employee give conflicting accounts.",
            ),
        ),
    ),
    FamilySpec(
        "disputes",
        ("billing correction", "service complaint", "delivery issue"),
        (
            "issue the standard low-value credit",
            "send the approved service-recovery response",
            "use the documented replacement process",
        ),
        (
            ("dispute_or_claim", "The customer threatens arbitration if the demand is not paid."),
            (
                "regulator_or_government_contact",
                "A consumer authority asks for the company's response.",
            ),
            (
                "material_external_commitment",
                "The proposed response admits fault and promises future payment.",
            ),
            ("outside_approved_playbook", "The demanded remedy exceeds the self-service limit."),
        ),
    ),
    FamilySpec(
        "corporate",
        ("routine entity record", "board calendar item", "standard insurance certificate"),
        (
            "update the approved administrative record",
            "place the recurring information item on the calendar",
            "send the current approved certificate",
        ),
        (
            (
                "material_external_commitment",
                "The document would guarantee another company's debt.",
            ),
            ("new_jurisdiction", "The team plans to register an entity in a new country."),
            (
                "regulator_or_government_contact",
                "A corporate registry issued a compulsory information request.",
            ),
            (
                "outside_approved_playbook",
                "The transaction type is not covered by the corporate checklist.",
            ),
        ),
    ),
    FamilySpec(
        "product",
        ("beta feature request", "API behavior change", "user-notification update"),
        (
            "use the approved beta terms without changes",
            "apply the documented backward-compatible change process",
            "send the approved operational notice",
        ),
        (
            ("legal_interpretation", "The team asks whether the feature creates a statutory duty."),
            ("sensitive_personal_data", "The feature infers a user's medical status."),
            ("new_jurisdiction", "The launch adds a country outside the approved release list."),
            ("outside_approved_playbook", "The feature has no completed product review playbook."),
        ),
    ),
    FamilySpec(
        "legal_operations",
        ("invoice coding request", "matter-file closure", "template access request"),
        (
            "use the published coding guide",
            "complete the standard retention checklist",
            "grant access through the approved role",
        ),
        (
            (
                "incomplete_or_conflicting_facts",
                "The requester cannot identify the matter or responsible team.",
            ),
            (
                "outside_approved_playbook",
                "The requested access role does not exist in the approved matrix.",
            ),
            (
                "sensitive_personal_data",
                "The files contain unredacted health and identity records.",
            ),
            ("dispute_or_claim", "The closed file has received a new threatened claim."),
        ),
    ),
)

COMPANIES = (
    "Aster Peak Systems",
    "Blue Lantern Labs",
    "Cedar Orbit Software",
    "Juniper Vale Robotics",
    "Northstar Software",
    "Redwood Signal Works",
)
REQUESTERS = (
    "Avery Chen",
    "Casey Morgan",
    "Jordan Patel",
    "Morgan Rivera",
    "Riley Thompson",
    "Taylor Brooks",
)
TEAMS = ("Finance", "People Operations", "Product", "Sales", "Security", "Support")
STYLES = ("email", "ticket", "chat", "intake_form", "summary")

SAFE_CLAUSES = {
    "train": (
        "All required facts are complete, and no exception is requested.",
        "The request stays within the documented limits and contains no special terms.",
        "The requester confirmed that the standard process fully resolves the matter.",
    ),
    "validation": (
        "The documented procedure applies without deviation and fully addresses the request.",
        "Intake is complete, and the requester identified no exception to the approved process.",
        "The request fits every condition in the published self-service instructions.",
    ),
    "test": (
        "The owner verified that the approved workflow resolves the request as submitted.",
        "No individualized judgment or departure from the standard process is requested.",
        "The submitted facts are complete and remain inside the self-service boundary.",
    ),
    "challenge": (
        "This is not a claim, incident, special term, new commitment, or request for an exception.",
        "The unusual deadline does not change the complete, standard, approved process.",
        "No hidden exception was identified; the ordinary self-service route fully resolves it.",
    ),
}

ATTENTION_WRAPPERS = {
    "train": "{clause}",
    "validation": "The intake includes one exception: {clause}",
    "test": "A later update adds a fact outside the routine process: {clause}",
    "challenge": "The request was called standard, but that description is incomplete. {clause}",
}

INDIRECT_TRIGGER_CLAUSES = {
    "legal_interpretation": (
        "The business needs a judgment about what a legal requirement means for these facts.",
        "The requested answer depends on interpreting an obligation rather than "
        "following a checklist.",
        "The owner is asking how a rule applies to this specific situation.",
    ),
    "nonstandard_contract_terms": (
        "The latest paper no longer matches the approved fallback positions.",
        "A risk provision was changed after the standard version was selected.",
        "The counterparty's draft contains language outside the accepted clause library.",
    ),
    "dispute_or_claim": (
        "Another party says it will pursue a formal remedy unless its demand is accepted.",
        "The correspondence now asserts wrongdoing and requests compensation.",
        "What began as an operational complaint has become a threatened formal proceeding.",
    ),
    "regulator_or_government_contact": (
        "An authority with compulsory powers has requested information.",
        "The company received an official inquiry carrying a response deadline.",
        "A public agency contacted the team about the underlying activity.",
    ),
    "sensitive_personal_data": (
        "The proposed record reveals health, identity, or similarly sensitive information.",
        "The workflow would expose a highly sensitive attribute about an identifiable person.",
        "The request includes data that receives heightened protection.",
    ),
    "security_incident": (
        "Available evidence suggests that an unauthorized person may have entered the system.",
        "The team is investigating possible access to information by an unknown party.",
        "A suspected compromise is part of the request, even though the scope is not settled.",
    ),
    "employment_action": (
        "The manager wants an individualized decision affecting one person's employment.",
        "The proposed step would materially change or end a specific worker's role.",
        "The request concerns a personnel action tied to one employee's conduct.",
    ),
    "intellectual_property_ownership": (
        "The parties do not agree about who owns the relevant work.",
        "The chain of title for the deliverable is contested or unclear.",
        "A contributor claims rights inconsistent with the planned use.",
    ),
    "new_jurisdiction": (
        "The activity would extend beyond every country currently covered by the playbook.",
        "The team is entering a territory that has not been reviewed for this process.",
        "The request adds a country outside the approved operating list.",
    ),
    "material_external_commitment": (
        "The proposed response would bind the company to a meaningful new promise.",
        "The team wants to guarantee an outcome that is not already approved.",
        "Proceeding would create an externally enforceable commitment outside the standard terms.",
    ),
    "incomplete_or_conflicting_facts": (
        "Two sources disagree about a fact that changes the routing decision.",
        "The requester cannot supply information required by the checklist.",
        "A material part of the intake remains unknown or internally inconsistent.",
    ),
    "outside_approved_playbook": (
        "No published self-service route covers this version of the request.",
        "The selected process does not contain an approved path for these facts.",
        "The request falls beyond the documented workflow boundary.",
    ),
}

NEGATED_RISK_CLAUSES = (
    ("dispute_or_claim", "No party has asserted a claim or threatened a formal proceeding."),
    (
        "regulator_or_government_contact",
        "No regulator, court, law-enforcement body, or government authority has "
        "contacted the team.",
    ),
    ("security_incident", "There is no known or suspected security incident."),
    (
        "sensitive_personal_data",
        "The ordinary operational details do not contain health, identity, or other "
        "sensitive data.",
    ),
    (
        "nonstandard_contract_terms",
        "No contract language or other negotiated term is being changed.",
    ),
    (
        "material_external_commitment",
        "The request does not add a guarantee, admission, public promise, or payment commitment.",
    ),
    (
        "employment_action",
        "No hiring, discipline, termination, accommodation, or other individual "
        "personnel action is involved.",
    ),
    (
        "intellectual_property_ownership",
        "No one disputes ownership of the relevant work.",
    ),
    ("new_jurisdiction", "The activity remains within the already approved countries."),
)

DISTRACTORS = (
    "The internal target date is next Thursday.",
    "The requester prefers a response before the monthly planning meeting.",
    "The project codename is Aurora Finch.",
    "The request was discussed during a routine status meeting.",
)


def _render(
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
        f"{company}'s {team} team wants to {action} concerning {subject}. "
        f"{policy_clause} {negated_risks} {distractor}"
    )
    if style == "email":
        if split == "validation":
            return (
                f"To: Request routing\nFrom: {requester}\n"
                f"Re: {subject} ({reference})\n\nContext supplied by {team}: {body}"
            )
        if split == "test":
            return (
                f"Email excerpt {reference}\nSender: {requester}\nThe {team} owner writes: {body}"
            )
        if split == "challenge":
            return (
                f"Forwarded note {reference}\nOriginally marked routine: yes\n"
                f"Author: {requester}\nDetails: {body}"
            )
        return (
            f"Subject: Request about {subject} [{reference}]\n\nHi team,\n\n"
            f"{body}\n\nThanks,\n{requester}"
        )
    if style == "ticket":
        if split == "validation":
            return (
                f"Case {reference}\nSubmitted by: {requester}\nOwning group: {team}\n"
                f"Intake narrative: {body}\nRouting requested: yes"
            )
        if split == "test":
            return (
                f"Queue item: {reference}\nBusiness contact: {requester}\n"
                f"Operational team: {team}\nFacts provided: {body}"
            )
        if split == "challenge":
            return (
                f"Ticket {reference}\nSelf-service selected by requester: yes\n"
                f"Requester: {requester}\nFull record: {body}"
            )
        return (
            f"Reference: {reference}\nRequester: {requester}\nTeam: {team}\n"
            f"Request: {body}\nStatus: Awaiting routing"
        )
    if style == "chat":
        if split == "validation":
            return (
                f"{requester} [{team}]: Routing check for {reference}. Full context follows. {body}"
            )
        if split == "test":
            return (
                f"Channel transcript, {reference}. {requester}: Please classify this intake. {body}"
            )
        if split == "challenge":
            return (
                f"{requester}: I selected the routine option for {reference}. "
                f"Please use the complete facts, though: {body}"
            )
        return f"{requester}: Can someone route {reference}? {body}"
    if style == "intake_form":
        if split == "validation":
            return (
                f"Intake ID: {reference}\nSubmitter: {requester}\nFunction: {team}\n"
                f"Complete description: {body}\nRequested classification: attention or self-service"
            )
        if split == "test":
            return (
                f"Form reference: {reference}\nOwner name: {requester}\nOwner team: {team}\n"
                f"Facts for routing: {body}\nSubmission complete: yes"
            )
        if split == "challenge":
            return (
                f"Intake {reference}\nRequester chose standard workflow: yes\n"
                f"Business owner: {requester}\nTeam: {team}\nDo not rely on the checkbox alone.\n"
                f"Narrative: {body}"
            )
        return (
            f"Reference: {reference}\nCompany: {company}\n"
            f"Business owner: {requester}\nArea: {team}\n"
            f"Request summary: {body}\nRequested outcome: Proceed or escalate"
        )
    if split == "validation":
        return f"Routing memorandum {reference}. Prepared for {team} from {requester}. {body}"
    if split == "test":
        return f"Condensed intake {reference}; business owner {requester}, team {team}. {body}"
    if split == "challenge":
        return (
            f"Summary {reference}. The header says routine. The narrative controls. "
            f"Prepared by {requester}. {body}"
        )
    return f"Request summary {reference}, prepared by {requester}. {body}"


def _seed_for(seed: int, split: str, pair_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{split}:{pair_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _template_id(
    split: str,
    family: str,
    style: str,
    trigger: str,
    trigger_variant: int,
    negated_risk_count: int,
) -> str:
    return f"{split}:{family}:{style}:{trigger}:v{trigger_variant}:n{negated_risk_count}"


def generate_split(split: str, size: int, seed: int) -> list[DecisionRecord]:
    """Generate one balanced split as counterfactual pairs."""
    if size % 2:
        raise ValueError("split size must be even")
    records: list[DecisionRecord] = []
    for pair_index in range(size // 2):
        rng = random.Random(_seed_for(seed, split, pair_index))
        family = FAMILIES[pair_index % len(FAMILIES)]
        style = STYLES[(pair_index // len(FAMILIES)) % len(STYLES)]
        trigger_field, family_trigger_clause = family.trigger_clauses[
            (pair_index // (len(FAMILIES) * len(STYLES))) % len(family.trigger_clauses)
        ]
        company = rng.choice(COMPANIES)
        requester = rng.choice(REQUESTERS)
        team = rng.choice(TEAMS)
        subject = rng.choice(family.subjects)
        action = rng.choice(family.actions)
        distractor = rng.choice(DISTRACTORS)
        safe_clause = rng.choice(SAFE_CLAUSES[split])
        pair_id = f"{split}-{pair_index:04d}"
        reference = f"NS-{split[:3].upper()}-{pair_index:04d}"
        trigger_options = (family_trigger_clause, *INDIRECT_TRIGGER_CLAUSES[trigger_field])
        trigger_variant = rng.randrange(len(trigger_options))
        trigger_clause = trigger_options[trigger_variant]
        attention_clause = ATTENTION_WRAPPERS[split].format(clause=trigger_clause)
        negated_risk_count = {
            "train": pair_index % 2,
            "validation": 1,
            "test": 1 + pair_index % 2,
            "challenge": 2 + pair_index % 2,
        }[split]
        available_negated_risks = [
            clause for field, clause in NEGATED_RISK_CLAUSES if field != trigger_field
        ]
        negated_risks = " ".join(rng.sample(available_negated_risks, k=negated_risk_count))
        template_id = _template_id(
            split,
            family.name,
            style,
            trigger_field,
            trigger_variant,
            negated_risk_count,
        )

        safe_facts = ScenarioFacts()
        attention_facts = replace(safe_facts, **{trigger_field: True})
        variants = (
            ("a", safe_facts, safe_clause),
            ("b", attention_facts, attention_clause),
        )
        for suffix, facts, policy_clause in variants:
            decision = evaluate_policy(facts)
            records.append(
                DecisionRecord(
                    record_id=f"{pair_id}-{suffix}",
                    pair_id=pair_id,
                    split=split,
                    family=family.name,
                    style=style,
                    template_id=template_id,
                    text=_render(
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
                    ),
                    facts=facts.to_dict(),
                    label=decision.label,
                    reasons=decision.reasons,
                )
            )
    rng = random.Random(seed + sum(ord(char) for char in split))
    rng.shuffle(records)
    return records


def generate_dataset(output_dir: Path = DATA_DIR, seed: int = 20261002) -> dict[str, Path]:
    """Generate all reproducible dataset splits and training adapters."""
    paths: dict[str, Path] = {}
    for split, size in SPLIT_SIZES.items():
        records = generate_split(split, size, seed)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        paths[split] = path
    return paths
