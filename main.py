from fastapi import FastAPI
from prometheus_client import make_asgi_app
from api.inference import router as inference_router

app = FastAPI(
    title="Moroccan Music Transformer API",
    description="Music generation API with Prometheus monitoring",
    version="1.0.0"
)

# Routers
app.include_router(
    inference_router,
    prefix="/api",
    tags=["Inference"]
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
