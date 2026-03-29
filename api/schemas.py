from pydantic import BaseModel, Field, field_validator
from typing import Optional


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(
        ..., ge=0, le=20,
        description="Fixed acidity",
        examples=[7.4]
    )
    volatile_acidity: float = Field(
        ..., ge=0, le=2,
        description="Volatile acidity",
        examples=[0.7]
    )
    citric_acid: float = Field(
        ..., ge=0, le=2,
        description="Citric acid",
        examples=[0.0]
    )
    residual_sugar: float = Field(
        ..., ge=0, le=20,
        description="Residual sugar",
        examples=[1.9]
    )
    chlorides: float = Field(
        ..., ge=0, le=1,
        description="Chlorides",
        examples=[0.076]
    )
    free_sulfur_dioxide: float = Field(
        ..., ge=0, le=100,
        description="Free sulfur dioxide",
        examples=[11.0]
    )
    total_sulfur_dioxide: float = Field(
        ..., ge=0, le=400,
        description="Total sulfur dioxide",
        examples=[34.0]
    )
    density: float = Field(
        ..., ge=0.9, le=1.1,
        description="Density",
        examples=[0.9978]
    )
    ph: float = Field(
        ..., ge=2.0, le=5.0,
        alias="pH",
        description="pH",
        examples=[3.51]
    )
    sulphates: float = Field(
        ..., ge=0, le=3,
        description="Sulphates",
        examples=[0.56]
    )
    alcohol: float = Field(
        ..., ge=5, le=20,
        description="Alcohol",
        examples=[9.4]
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "fixed_acidity": 7.4,
                    "volatile_acidity": 0.7,
                    "citric_acid": 0.0,
                    "residual_sugar": 1.9,
                    "chlorides": 0.076,
                    "free_sulfur_dioxide": 11.0,
                    "total_sulfur_dioxide": 34.0,
                    "density": 0.9978,
                    "pH": 3.51,
                    "sulphates": 0.56,
                    "alcohol": 9.4,
                }
            ]
        },
    }

    def to_feature_list(self) -> list[float]:
        return [
            self.fixed_acidity,
            self.volatile_acidity,
            self.citric_acid,
            self.residual_sugar,
            self.chlorides,
            self.free_sulfur_dioxide,
            self.total_sulfur_dioxide,
            self.density,
            self.ph,
            self.sulphates,
            self.alcohol,
        ]

    def to_feature_names(self) -> list[str]:
        return [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ]


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted wine quality")
    model_name: str = Field(..., description="Name of the model used")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_params: dict
    training_date: Optional[str] = None
    metrics: Optional[dict] = None
    feature_columns: Optional[list[str]] = None
    data_rows: Optional[int] = None


class ErrorResponse(BaseModel):
    detail: str
