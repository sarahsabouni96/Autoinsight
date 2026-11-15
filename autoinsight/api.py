from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Query

from .inference import Car, Predictor
from .storage import History


def create_app(model_path=None, database_url=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = Predictor(model_path or os.getenv("MODEL_PATH", "artifacts/used_car_pipeline.pkl"))
        app.state.history = History(database_url or os.getenv("DATABASE_URL", "sqlite:///history.db"))
        try:
            yield
        finally:
            app.state.history.engine.dispose()

    app = FastAPI(title="AutoInsight", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready", "model_version": app.state.predictor.version}

    @app.get("/metadata")
    def metadata():
        return app.state.predictor.report

    @app.post("/predict")
    def predict(car: Car):
        try:
            result = app.state.predictor.predict(car)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        app.state.history.append(car.model_dump(), result)
        return result

    @app.get("/history")
    def history(limit: int = Query(30, ge=1, le=100)):
        return app.state.history.recent(limit)

    return app


app = create_app()
