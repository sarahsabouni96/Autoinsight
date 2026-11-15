from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/car_sales_data.csv")
LOG = ["Engine size", "Mileage"]
YEAR = "Year of manufacture"
CAT = ["Manufacturer", "Model", "Fuel type"]
FEATURES = LOG + [YEAR] + CAT
TARGET = "Price"


def read_sales(path=DATA):
    raw = pd.read_csv(path)
    cols = FEATURES + [TARGET]
    missing = set(cols) - set(raw)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = raw[cols].copy()
    for col in LOG + [YEAR, TARGET]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].dropna()).all():
            raise ValueError(f"Non-finite {col}")
    for col in CAT:
        df[col] = df[col].str.strip().replace("", np.nan)
    if df[TARGET].isna().any() or (df[TARGET] <= 0).any():
        raise ValueError("Expected positive, observed prices.")
    if (df["Engine size"] <= 0).any() or (df["Mileage"] < 0).any():
        raise ValueError("Invalid engine size or mileage.")
    years = df[YEAR].dropna()
    if ((years < 1886) | (years % 1 != 0)).any():
        raise ValueError("Invalid year.")
    unique = df.drop_duplicates()
    if unique.duplicated(FEATURES).any():
        raise ValueError("Conflicting prices for identical profiles. Review grouping before splitting.")
    return unique.reset_index(drop=True), len(raw) - len(unique)


def partitions(frame):
    from sklearn.model_selection import train_test_split

    development, test = train_test_split(frame, test_size=.2, random_state=42)
    fit, calibration = train_test_split(development, test_size=.2, random_state=42)
    return fit, calibration, test
