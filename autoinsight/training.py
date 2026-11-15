import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CAT, DATA, FEATURES, LOG, TARGET, YEAR, partitions, read_sales


def preprocessor():
    return ColumnTransformer([
        ("log_numeric", make_pipeline(SimpleImputer(strategy="median"),
                                     FunctionTransformer(np.log1p), StandardScaler()), LOG),
        ("year", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), [YEAR]),
        ("categories", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                     OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CAT),
    ])


def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "R2": float(r2_score(y, pred))}


def train(path=DATA, destination=None):
    df, duplicates = read_sales(path)
    fit, cal, test = partitions(df)
    X, y = fit[FEATURES], fit[TARGET]
    pipeline = Pipeline([("features", preprocessor()), ("model", Ridge())])
    cv = GridSearchCV(pipeline, [
        {"model": [Ridge()], "model__alpha": [1., 10.]},
        {"model": [HistGradientBoostingRegressor(max_iter=120, random_state=42)],
         "model__max_leaf_nodes": [15, 31]},
    ], cv=KFold(3, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    cv.fit(X, y)
    winner = cv.best_estimator_
    residuals = np.abs(cal[TARGET] - winner.predict(cal[FEATURES]))
    rank = math.ceil(.9 * (len(cal) + 1))
    radius = float(np.sort(residuals)[rank - 1])
    pred = winner.predict(test[FEATURES])
    baseline = DummyRegressor(strategy="median").fit(X, y)
    diagnostic = cal.sample(min(1000, len(cal)), random_state=42)
    importance = permutation_importance(winner, diagnostic[FEATURES], diagnostic[TARGET],
                                       scoring="neg_mean_absolute_error", n_repeats=3, random_state=42)
    segment_cols = [YEAR, "Mileage", "Engine size"]
    segmenter = make_pipeline(SimpleImputer(), StandardScaler(), KMeans(n_clusters=3, n_init=10, random_state=42))
    labels = segmenter.fit_predict(X[segment_cols])
    medians = fit.assign(segment=labels).groupby("segment")[segment_cols + [TARGET]].median()
    pairs = X[["Manufacturer", "Model"]].dropna().drop_duplicates().sort_values(["Manufacturer", "Model"])
    report = {
        "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__, "duplicates_removed": duplicates, "seed": 42,
        "rows": {"fit": len(fit), "calibration": len(cal), "test": len(test)},
        "model": type(winner.named_steps["model"]).__name__,
        "candidates": [{"settings": str(p), "cv_mae": float(-s)} for p, s in
                       zip(cv.cv_results_["params"], cv.cv_results_["mean_test_score"])],
        "test": metrics(test[TARGET], pred), "baseline": metrics(test[TARGET], baseline.predict(test[FEATURES])),
        "interval_radius": radius, "test_coverage": float((np.abs(test[TARGET] - pred) <= radius).mean()),
        "importance": dict(zip(FEATURES, importance.importances_mean.tolist())),
        "segment_medians": json.loads(medians.to_json(orient="index")),
        "models_by_make": pairs.groupby("Manufacturer")["Model"].apply(list).to_dict(),
        "fuels": sorted(X["Fuel type"].dropna().unique().tolist()),
        "make_medians": fit.groupby("Manufacturer")[TARGET].median().to_dict(),
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in segment_cols},
    }
    bundle = {"pipeline": winner, "segmenter": segmenter, "segment_cols": segment_cols, "report": report}
    dest = Path(destination or os.getenv("MODEL_PATH", "artifacts/used_car_pipeline.pkl"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        joblib.dump(bundle, temp, compress=3)
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
