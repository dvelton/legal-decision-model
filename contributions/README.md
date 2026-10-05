# Fictional case contributions

This project accepts only wholly fictional test cases. Do not submit real legal requests, contract
language, customer information, employee information, personal data, privileged material,
confidential facts, internal links, or text copied from a company system.

## Purpose

Contributions can improve behavioral coverage by adding fictional cases that test:

- negation and quoted text;
- misleading headings;
- facts revealed late in a long request;
- multiple escalation triggers;
- fact-preserving paraphrases;
- out-of-scope requests; or
- difficult counterfactual pairs.

## Submission format

Create a JSON file that satisfies `contributions/schema.json`. A contribution must include:

- a statement that the content is fictional and newly written;
- the proposed policy facts;
- a safe and attention-required counterfactual;
- the exact facts changed between the two records;
- two fact-preserving renderings of each record; and
- the behavioral capabilities being tested.

`review-queue.example.jsonl` shows the queue format. Its entry is pending and is not counted as
human-reviewed.

Open a pull request containing the JSON file and a short explanation of the intended test. Do not
add the case directly to the benchmark generator. A maintainer must review the structured facts,
deterministic label, originality statement, and public-content scan before incorporation.

Validate the file before opening the pull request:

```bash
uv run legal-decision-model contribution validate contributions/my-fictional-case.json
```

## Review status

A submitted case is not described as human-reviewed merely because a person opened a pull request.
The review record must identify the reviewed policy version and record one of these states:

- `pending`
- `accepted`
- `rejected`

Only an `accepted` case with a named reviewer and review date may be counted as human-reviewed.
