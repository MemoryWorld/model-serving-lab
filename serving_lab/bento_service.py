"""BentoML-hosted gateway; lifecycle and inference stay in the existing ASGI app."""

import bentoml

from serving_lab.app import create_app

# A dedicated prefix avoids confusing Bento's process readiness with backend readiness.
# BentoML runs mounted ASGI lifespans, including backend creation and shutdown cleanup.
gateway = create_app()


@bentoml.service(
    name="model_serving_lab",
    workers=1,
    resources={"cpu": "1", "memory": "512Mi"},
    traffic={"timeout": 70},
    logging={"access": {"enabled": False}},
)
@bentoml.asgi_app(gateway, path="/gateway")
class ModelServingLab:
    """Real Bento service packaging the shared OpenAI-compatible gateway.

    /gateway/readyz checks the configured model backend. /readyz is Bento's
    worker health only. Authentication/queueing/SSE are shared with FastAPI.
    """
