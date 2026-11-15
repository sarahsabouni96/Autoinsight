from datetime import date
import hashlib
from pathlib import Path

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data import FEATURES


class Car(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)
    manufacturer: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    fuel_type: str = Field(min_length=1, max_length=80)
    engine_size: float = Field(gt=0)
    year: int = Field(ge=1886)
    mileage: float = Field(ge=0)

    @field_validator("year")
    @classmethod
    def valid_year(cls, year):
        if year > date.today().year + 1:
            raise ValueError("Year is beyond the next model year.")
        return year


class Predictor:
    def __init__(self, path):
        path = Path(path)
        with path.open("rb") as stream:
            self.version = hashlib.file_digest(stream, "sha256").hexdigest()[:16]
            stream.seek(0)
            self.bundle = joblib.load(stream)
        self.report = self.bundle["report"]

    def predict(self, car):
        models = self.report["models_by_make"].get(car.manufacturer, [])
        if car.model not in models or car.fuel_type not in self.report["fuels"]:
            raise ValueError("Choose a supported manufacturer/model pair and fuel type.")
        record = {"Manufacturer": car.manufacturer, "Model": car.model, "Fuel type": car.fuel_type,
                  "Year of manufacture": car.year, "Engine size": car.engine_size, "Mileage": car.mileage}
        for name, (low, high) in self.report["ranges"].items():
            if not low <= record[name] <= high:
                raise ValueError(f"{name} must be within the observed training range {low:g}–{high:g}.")
        x = pd.DataFrame([record], columns=FEATURES)
        price = float(self.bundle["pipeline"].predict(x)[0])
        radius = self.report["interval_radius"]
        segment = int(self.bundle["segmenter"].predict(x[self.bundle["segment_cols"]])[0])
        return {"estimate": round(price, 2), "lower": round(max(0, price - radius), 2),
                "upper": round(price + radius, 2), "segment": segment, "model_version": self.version}
