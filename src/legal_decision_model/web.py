"""Local demonstration web application."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from legal_decision_model.inference import Decision, LegalDecisionModel


class DecisionRequest(BaseModel):
    """One request submitted to the demonstration model."""

    text: str = Field(min_length=1, max_length=8_000)


class DecisionResponse(BaseModel):
    """Typed API response."""

    decision: str
    requires_human_lawyer_attention: bool
    probability_no_human_lawyer_attention: float
    probability_requires_human_lawyer_attention: float
    clearance_threshold: float | None
    routed_conservatively: bool
    routing_reason: str
    latency_ms: int
    model: str
    policy_version: str
    disclaimer: str


class DecisionModel(Protocol):
    """Inference contract used by the web application."""

    def predict(self, text: str) -> Decision: ...


def create_app(
    model_factory: Callable[[], DecisionModel] = LegalDecisionModel,
) -> FastAPI:
    """Create the local-only demonstration API."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = model_factory()
        yield

    app = FastAPI(
        title="Legal Decision Model",
        version="0.1.0",
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
    document.querySelector("#metrics").textContent =
      `Safe-clear probability: ${(payload.probability_no_human_lawyer_attention * 100).toFixed(1)}%`
      + ` | Required threshold: ${payload.clearance_threshold === null
        ? "clearance disabled"
        : (payload.clearance_threshold * 100).toFixed(1) + "%"}`
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
</script>
</body>
</html>
"""
