import os
import numpy as np

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autoinsight.api import create_app
from autoinsight.data import DATA, FEATURES, partitions, read_sales
from autoinsight.training import train


@pytest.mark.skipif(not DATA.exists(), reason="Original CSV supplied separately")
def test_source():
    df, duplicates = read_sales()
    assert len(df) == 49988
    assert duplicates == 12
    assert not df.duplicated(FEATURES).any()
    fit, cal, test = partitions(df)
    assert [len(fit), len(cal), len(test)] == [31992, 7998, 9998]
    assert fit.index.intersection(test.index).empty
    assert cal.index.intersection(test.index).empty
    assert fit.index.intersection(cal.index).empty


def sample_data():
    if DATA.exists():
        return pd.read_csv(DATA).sample(900, random_state=9)
    rng = np.random.default_rng(9)
    year = rng.integers(2000, 2023, 900)
    mileage = rng.choice(190000, 900, replace=False)
    return pd.DataFrame({"Manufacturer": "Ford", "Model": "Fiesta", "Fuel type": "Petrol",
                         "Mileage": mileage, "Year of manufacture": year,
                         "Engine size": rng.choice([1., 1.6, 2.], 900),
                         "Price": 18000 + (year - 2000) * 500 - mileage * .04})


def test_bad_price(tmp_path):
    raw = sample_data().head(4).copy()
    raw.loc[0, "Price"] = -1
    path = tmp_path / "bad.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="prices"):
        read_sales(path)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    folder = tmp_path_factory.mktemp("training")
    csv = folder / "sales.csv"
    model = folder / "pipeline.pkl"
    sample_data().to_csv(csv, index=False)
    report = train(csv, model)
    assert sum(report["rows"].values()) + report["duplicates_removed"] == 900
    return model


def test_prediction_and_history(fitted, tmp_path):
    db = os.getenv("TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'history.db'}")
    car = dict(manufacturer="Ford", model="Fiesta", year=2015, mileage=50000, engine_size=1.6, fuel_type="Petrol")
    with TestClient(create_app(fitted, db)) as client:
        assert client.get("/health").status_code == 200
        assert "test" in client.get("/metadata").json()
        result = client.post("/predict", json=car)
        assert result.status_code == 200, result.text
        result = result.json()
        assert result["lower"] <= result["estimate"] <= result["upper"]
        row = client.get("/history?limit=1").json()[0]
        assert row["car"] == car and row["result"] == result
        for invalid in ({"mileage": -1}, {"model": "Unknown"}, {"fuel_type": ""}, {"Price": 100}):
            assert client.post("/predict", json={**car, **invalid}).status_code == 422
        assert client.get("/history?limit=1").json()[0]["id"] == row["id"]
    with TestClient(create_app(fitted, db)) as client:
        assert client.get("/history?limit=1").json()[0]["result"] == result
