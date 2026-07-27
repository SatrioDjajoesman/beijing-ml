#!/usr/bin/env python3
"""
Stage A: unsupervised leak anomaly detection.

Trains one IsolationForest per (device, position) using the engineered
features table, so each of the three regions (start/middle/end) on each
pipe gets an independent "is this region behaving abnormally" score.
Needs no labeled leak data — this is the baseline detector that runs
before enough real leak trials exist to train a supervised classifier
(Stage B) and time-to-leak estimator (Stage C).

Usage:
    ../esp32/venv/bin/python3 anomaly_detection.py [--db PATH] [--out anomaly_scores.csv]
"""

import argparse
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from db_path import default_db_path

DEFAULT_DB_PATH = default_db_path()
MODELS_DIR = Path(__file__).parent / "models"
POSITIONS = ["start", "middle", "end"]

# Cross-position pressure deltas relevant to each region — a region's
# model should see the gradients on either side of it, not just its own
# sensor.
SHARED_DELTA_COLUMNS = {
    "start": ["pressure_start_minus_middle", "pressure_start_minus_end"],
    "middle": ["pressure_start_minus_middle", "pressure_middle_minus_end"],
    "end": ["pressure_middle_minus_end", "pressure_start_minus_end"],
}


def load_features(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM features", conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def position_feature_columns(df: pd.DataFrame, position: str) -> list[str]:
    own = [c for c in df.columns if f"_{position}_" in c and not c.endswith("_status")]
    shared = [c for c in SHARED_DELTA_COLUMNS.get(position, []) if c in df.columns]
    # A delta column like "pressure_start_minus_middle" already matches the
    # "_start_" substring, so `own` and `shared` can overlap — dedupe while
    # preserving order (duplicate columns would otherwise compound under
    # repeated pandas indexing below and desync the model's feature count
    # from the saved column list).
    seen = set()
    columns = []
    for c in own + shared:
        if c not in seen:
            seen.add(c)
            columns.append(c)
    return columns


def score_position(df: pd.DataFrame, device_id: str, position: str):
    cols = position_feature_columns(df, position)
    if not cols:
        return None

    sub = df[["ts"] + cols].dropna()
    if len(sub) < 20:
        # Not enough clean history yet to fit a meaningful model.
        return None

    X = sub[cols].values
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(X)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": cols}, MODELS_DIR / f"isoforest_{device_id}_{position}.joblib")

    scores = model.decision_function(X)  # higher = more normal
    preds = model.predict(X)  # -1 = anomaly, 1 = normal

    result = sub[["ts"]].copy()
    result["device_id"] = device_id
    result["position"] = position
    result["anomaly_score"] = -scores  # flip sign: higher = more anomalous
    result["is_anomaly"] = preds == -1
    return result


def run(db_path: Path) -> pd.DataFrame:
    df = load_features(db_path)
    if df.empty:
        return pd.DataFrame()

    results = []
    for device_id, g in df.groupby("device_id"):
        for position in POSITIONS:
            r = score_position(g, device_id, position)
            if r is not None:
                results.append(r)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True).sort_values(["device_id", "position", "ts"])


def save_results(df: pd.DataFrame, db_path: Path, csv_path: Path | None = None):
    conn = sqlite3.connect(db_path)
    df.to_sql("anomaly_scores", conn, if_exists="replace", index=False)
    conn.close()

    if csv_path:
        df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to sensor_data.db")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "anomaly_scores.csv", help="CSV output path")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"No database found at {args.db}. Run feature_engineering.py first.")

    results = run(args.db)
    if results.empty:
        raise SystemExit("Not enough feature history yet to train anomaly models (need 20+ clean rows per region).")

    save_results(results, args.db, args.out)

    anomalies = results[results["is_anomaly"]]
    print(f"Scored {len(results)} region-timestamps across {results['device_id'].nunique()} device(s).")
    print(f"Flagged {len(anomalies)} anomalous readings.")
    if not anomalies.empty:
        latest = anomalies.sort_values("ts").groupby(["device_id", "position"]).tail(1)
        print("\nMost recent anomaly per region:")
        print(latest[["device_id", "position", "ts", "anomaly_score"]].to_string(index=False))

    print(f"\nSaved to table 'anomaly_scores' in {args.db} and CSV at {args.out}")
    print(f"Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
