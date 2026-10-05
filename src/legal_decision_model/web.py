"""Local demonstration web application."""

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from legal_decision_model.constants import POLICY_MODEL_DIR, PROJECT_ROOT
from legal_decision_model.inference import Decision, LegalDecisionModel
from legal_decision_model.policy_inference import PolicyDecision, PolicyDecisionModel


class DecisionRequest(BaseModel):
    """One request submitted to the demonstration model."""

    text: str = Field(min_length=1, max_length=200_000)


class DecisionResponse(BaseModel):
    """Typed API response."""

    decision: str
    requires_human_lawyer_attention: bool
    probability_no_human_lawyer_attention: float | None = None
    probability_requires_human_lawyer_attention: float | None = None
    clearance_score: float | None = None
    clearance_threshold: float | None
    workflow_family: str | None = None
    reason_codes: tuple[str, ...] = ()
    task_probabilities: dict[str, float] = Field(default_factory=dict)
    ensemble_disagreement: float | None = None
    ood_distance: float | None = None
    routed_conservatively: bool
    routing_reason: str
    token_count: int | None = None
    window_count: int | None = None
    latency_ms: int
    model: str
    policy_version: str
    disclaimer: str


class DecisionModel(Protocol):
    """Inference contract used by the web application."""

    def predict(self, text: str) -> Decision | PolicyDecision: ...


def default_model_factory() -> DecisionModel:
    """Prefer the improved policy ensemble when its artifact is installed."""
    if (POLICY_MODEL_DIR / "metadata.json").is_file():
        return PolicyDecisionModel()
    return LegalDecisionModel()


def create_app(
    model_factory: Callable[[], DecisionModel] = default_model_factory,
) -> FastAPI:
    """Create the local-only demonstration API."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = model_factory()
        yield

    app = FastAPI(
        title="Legal Decision Model",
        version="0.2.0",
        description=(
            "Synthetic proof of concept that routes a fictional request either to human lawyer "
            "attention or to a fictional approved self-service process."
        ),
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return _HTML

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/model")
    def model_info() -> dict[str, object]:
        metadata = app.state.model.metadata
        if hasattr(metadata, "ensemble_size"):
            return {
                "generation": "policy-v2",
                "model": metadata.base_model,
                "policy_version": metadata.policy_version,
                "ensemble_size": metadata.ensemble_size,
                "training_rows": metadata.training_rows,
                "validation_rows": metadata.validation_rows,
                "global_threshold": metadata.global_threshold,
                "maximum_false_clear_rate": metadata.maximum_false_clear_rate,
                "confidence_level": metadata.confidence_level,
                "max_tokens": metadata.max_tokens,
                "maximum_windows": metadata.maximum_windows,
            }
        return {
            "generation": "legacy-v1",
            "model": metadata.base_model,
            "policy_version": metadata.policy_version,
            "ensemble_size": 1,
            "training_rows": metadata.training_rows,
            "validation_rows": metadata.validation_rows,
            "global_threshold": metadata.clearance_threshold,
            "maximum_false_clear_rate": metadata.max_false_clear_rate,
            "max_tokens": metadata.max_len,
            "maximum_windows": 1,
        }

    @app.get("/v1/evaluation")
    def evaluation() -> dict[str, object]:
        model = app.state.model
        metadata = model.metadata
        if not hasattr(metadata, "ensemble_size") or not hasattr(model, "model_dir"):
            return {"available": False, "reports": {}}
        fingerprint = model.artifact_fingerprint
        directory = PROJECT_ROOT / "evaluation" / "policy-v2"
        manifest_path = directory / "benchmark-manifest.json"
        if not manifest_path.is_file():
            return {"available": False, "reports": {}}
        manifest = _read_json_object(manifest_path)
        split_hashes = manifest.get("split_sha256")
        seed = manifest.get("seed")
        if not isinstance(split_hashes, dict) or not isinstance(seed, int):
            return {"available": False, "reports": {}}
        with manifest_path.open("rb") as handle:
            manifest_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        reports = {}
        for split in ("test", "challenge"):
            path = directory / f"{split}.json"
            if path.is_file():
                report = _read_json_object(path)
                if (
                    report.get("artifact_fingerprint") == fingerprint
                    and report.get("policy_version") == metadata.policy_version
                    and report.get("base_model_revision") == metadata.base_model_revision
                    and report.get("benchmark_split_sha256") == split_hashes.get(split)
                    and report.get("benchmark_manifest_sha256") == manifest_sha256
                    and report.get("benchmark_seed") == seed
                ):
                    reports[split] = report
        return {
            "available": bool(reports),
            "reports": reports,
        }

    @app.post("/v1/decision", response_model=DecisionResponse)
    def make_decision(
        request: Annotated[DecisionRequest, Body()],
    ) -> DecisionResponse:
        try:
            result = app.state.model.predict(request.text)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return DecisionResponse.model_validate(result.to_dict())

    return app


def _read_json_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object")
    return raw


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Legal Decision Model</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #f4f5f7; color: #172033; }
    main { max-width: 880px; margin: 0 auto; padding: 48px 20px 72px; }
    h1 { margin-bottom: 8px; font-size: clamp(2rem, 5vw, 3.4rem); letter-spacing: -0.04em; }
    .lede { max-width: 720px; color: #4d5870; line-height: 1.6; }
    .card { margin-top: 28px; padding: 24px; background: white; border: 1px solid #d9deea;
      border-radius: 16px; box-shadow: 0 10px 30px rgba(25, 35, 60, 0.08); }
    textarea { box-sizing: border-box; width: 100%; min-height: 210px; padding: 16px;
      border: 1px solid #aab3c5; border-radius: 10px; resize: vertical; font: inherit;
      line-height: 1.5; }
    button { margin-top: 14px; padding: 11px 18px; border: 0; border-radius: 9px;
      color: white; background: #2548d8; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.55; cursor: wait; }
    .samples { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    .samples button { margin: 0; padding: 7px 10px; background: #e8ecfb; color: #243868;
      font-weight: 600; }
    #result { display: none; margin-top: 20px; padding: 18px; border-radius: 10px;
      background: #f2f5fb; }
    #result.attention { border-left: 5px solid #9d2a2a; }
    #result.clear { border-left: 5px solid #237447; }
    .decision { font-size: 1.2rem; font-weight: 800; overflow-wrap: anywhere; }
    .metrics { margin-top: 10px; color: #4d5870; line-height: 1.7; }
    .notice { margin-top: 20px; padding: 14px 16px; background: #fff7dd; border-radius: 9px;
      color: #5b4911; line-height: 1.5; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px; margin-top: 18px; }
    .metric { padding: 16px; border: 1px solid #d9deea; border-radius: 10px; background: #fafbfe; }
    .metric .value { display: block; margin-top: 6px; font-size: 1.3rem; font-weight: 800; }
    .small { color: #5c667a; font-size: 0.92rem; line-height: 1.5; }
    code { font-size: 0.9em; }
  </style>
</head>
<body>
<main>
  <h1>Legal Decision Model</h1>
  <p class="lede">A synthetic proof of concept with one output: whether a fictional corporate
  request requires human lawyer attention. Uncertainty is routed to human attention.</p>
  <section class="card">
    <label for="request"><strong>Fictional request</strong></label>
    <div class="samples">
      <button type="button" data-sample="safe">Load self-service example</button>
      <button type="button" data-sample="attention">Load escalation example</button>
    </div>
    <textarea id="request" placeholder="Paste or type a fictional request."></textarea>
    <button id="submit" type="button">Make decision</button>
    <div id="result" aria-live="polite">
      <div class="decision" id="decision"></div>
      <div class="metrics" id="metrics"></div>
    </div>
    <div class="notice">This demonstration was trained only on generated fictional data. It is
    not legal advice, is not validated for real matters, and must not be used to bypass legal,
    compliance, privacy, security, employment, or regulatory review.</div>
  </section>
  <section class="card">
    <h2>Model and evaluation</h2>
    <p class="small">Operating controls and held-out synthetic results from the installed artifact.
    Statistical bounds apply only to the generated evaluation population.</p>
    <div class="grid" id="model-grid"></div>
    <div class="grid" id="evaluation-grid"></div>
  </section>
</main>
<script>
const samples = {
  safe: `Condensed intake NS-TES-0444; business owner Morgan Rivera, team Support.
Redwood Signal Works's Support team wants to publish the already approved wording unchanged
concerning promotional giveaway. The owner verified that the approved workflow resolves the
request as submitted. There is no known or suspected security incident. The project codename is
Aurora Finch.`,
  attention: `Requester: Avery Chen
Team: Finance
Request: Northstar Software's Finance team wants to use the pre-approved template without edits
for a software subscription. The counterparty inserted a new unlimited liability clause.
The internal target date is next Thursday.
Status: Awaiting routing`
};
const textarea = document.querySelector("#request");
document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => { textarea.value = samples[button.dataset.sample]; });
});
document.querySelector("#submit").addEventListener("click", async () => {
  const submit = document.querySelector("#submit");
  const result = document.querySelector("#result");
  submit.disabled = true;
  submit.textContent = "Running...";
  try {
    const response = await fetch("/v1/decision", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({text: textarea.value})
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Request failed");
    result.style.display = "block";
    result.className = payload.requires_human_lawyer_attention ? "attention" : "clear";
    document.querySelector("#decision").textContent = payload.decision;
    const score = payload.clearance_score ?? payload.probability_no_human_lawyer_attention;
    const family = payload.workflow_family ? ` | Family: ${payload.workflow_family}` : "";
    const windows = payload.window_count ? ` | Windows: ${payload.window_count}` : "";
    document.querySelector("#metrics").textContent =
      `Clearance score: ${(score * 100).toFixed(1)}%`
      + ` | Required threshold: ${payload.clearance_threshold === null
        ? "clearance disabled"
        : (payload.clearance_threshold * 100).toFixed(1) + "%"}`
      + ` | Route: ${payload.routing_reason}`
      + family
      + windows
      + ` | ${payload.latency_ms} ms`;
  } catch (error) {
    result.style.display = "block";
    result.className = "attention";
    document.querySelector("#decision").textContent = "Unable to make a model decision";
    document.querySelector("#metrics").textContent = error.message;
  } finally {
    submit.disabled = false;
    submit.textContent = "Make decision";
  }
});
function metric(label, value) {
  const card = document.createElement("div");
  card.className = "metric";
  const name = document.createElement("span");
  name.textContent = label;
  const result = document.createElement("span");
  result.className = "value";
  result.textContent = value;
  card.append(name, result);
  return card;
}
async function loadDashboard() {
  const model = await fetch("/v1/model").then((response) => response.json());
  const modelGrid = document.querySelector("#model-grid");
  modelGrid.append(
    metric("Model generation", model.generation),
    metric("Policy", model.policy_version),
    metric("Ensemble heads", String(model.ensemble_size)),
    metric("Training records", Number(model.training_rows).toLocaleString()),
    metric("Validation records", Number(model.validation_rows).toLocaleString()),
    metric("Global risk limit", `${(model.maximum_false_clear_rate * 100).toFixed(2)}%`)
  );
  const evaluation = await fetch("/v1/evaluation").then((response) => response.json());
  const evaluationGrid = document.querySelector("#evaluation-grid");
  if (!evaluation.available) {
    evaluationGrid.append(metric("Held-out evaluation", "Run policy evaluate"));
    return;
  }
  for (const split of ["test", "challenge"]) {
    const report = evaluation.reports[split];
    if (!report) continue;
    const metrics = report.metrics;
    evaluationGrid.append(
      metric(`${split} attention recall`, `${(metrics.attention_recall * 100).toFixed(2)}%`),
      metric(`${split} clearance coverage`, `${(metrics.clearance_coverage * 100).toFixed(2)}%`),
      metric(`${split} false clears`, String(metrics.false_clearances)),
      metric(`${split} risk upper bound`, `${(metrics.upper_false_clear_rate * 100).toFixed(3)}%`),
      metric(`${split} pair consistency`,
        `${(report.counterfactual_pair_consistency * 100).toFixed(2)}%`),
      metric(`${split} paraphrase invariance`,
        `${(report.paraphrase_invariance * 100).toFixed(2)}%`)
    );
  }
}
loadDashboard().catch(() => {
  document.querySelector("#evaluation-grid").append(metric("Dashboard", "Unavailable"));
});
</script>
</body>
</html>
"""
