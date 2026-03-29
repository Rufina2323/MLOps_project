import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import Request
import uvicorn

from schemas import (
    WineFeatures,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from model_loader import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """При старте приложения загружаем модель из DVC."""
    logger.info("Starting up — loading model...")
    try:
        model_manager.load()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine quality using a Ridge regression model. "
                "Model is loaded from DVC on startup.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error["loc"])
        errors.append(f"{field}: {error['msg']}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": errors,
        },
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"description": "Validation error"},
    },
)
def predict(data: WineFeatures) -> PredictionResponse:
    """
    Предсказание качества вина.

    Принимает характеристики вина, возвращает предсказанную оценку качества.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health for status.",
        )

    try:
        features = data.to_feature_list()
        prediction = model_manager.predict(features)
        return PredictionResponse(
            prediction=round(prediction, 4),
            model_name=model_manager.metadata.get("model_name", "unknown"),
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Проверка работоспособности сервиса."""
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
def model_info() -> ModelInfoResponse:
    """Информация о модели и её метриках."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    info = model_manager.get_info()
    return ModelInfoResponse(**info)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)