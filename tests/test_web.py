import asyncio

import httpx

from legal_decision_model.constants import REQUIRES_ATTENTION
from legal_decision_model.inference import Decision
from legal_decision_model.web import create_app


class FakeModel:
    def predict(self, text: str) -> Decision:
        assert text == "fictional request"
        return Decision(
            decision=REQUIRES_ATTENTION,
            requires_human_lawyer_attention=True,
            probability_no_human_lawyer_attention=0.2,
            probability_requires_human_lawyer_attention=0.8,
            clearance_threshold=0.9,
            routed_conservatively=False,
            routing_reason="MODEL_DECISION",
            latency_ms=12,
            model="test-model",
            policy_version="northstar-1.0",
        )


def test_decision_api() -> None:
    app = create_app(model_factory=FakeModel)

    async def request() -> httpx.Response:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post("/v1/decision", json={"text": "fictional request"})

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["decision"] == REQUIRES_ATTENTION


def test_model_factory_runs_once_at_startup() -> None:
    calls = 0

    def factory() -> FakeModel:
        nonlocal calls
        calls += 1
        return FakeModel()

    app = create_app(model_factory=factory)

    async def startup() -> None:
        async with app.router.lifespan_context(app):
            assert calls == 1

    asyncio.run(startup())
    assert calls == 1
