#!/usr/bin/env python3
"""
Stage C: time-to-leak estimation via trend extrapolation.

Takes the per-region anomaly scores from anomaly_detection.py, fits a
local linear trend to each region's recent score trajectory, and
extrapolates when it will cross the IsolationForest's own decision
boundary (anomaly_score = 0). This is a simple, explainable estimate —
not a learned survival model — appropriate before enough labeled
run-to-failure trials exist to train one.

Usage:
    ../esp32/venv/bin/python3 time_to_leak.py [--db PATH] [--out estimates.csv]
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).parent.parent / "esp32" / "sensor_data.db"

WINDOW_POINTS = 20  # how many recent samples to fit the trend on
MIN_POINTS = 5  # minimum samples required to attempt a fit
MAX_HORIZON_SECONDS = 6 * 3600  # beyond this, a linear extrapolation isn't trustworthy — report "stable"
ANOMALY_THRESHOLD = 0.0  # matches the IsolationForest decision boundary


def load_table(db_path: Path, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except pd.errors.DatabaseError:
        return pd.DataFrame()
    finally:
        conn.close()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def fit_trend(times: pd.Series, scores: pd.Series):
    """Least-squares linear fit of score vs. seconds elapsed within the
    window. Returns (slope_per_second, score_at_latest_point)."""
    t_seconds = (times - times.iloc[0]).dt.total_seconds().values
    slope, intercept = np.polyfit(t_seconds, scores.values, 1)
    score_now = slope * t_seconds[-1] + intercept
    return slope, score_now


def estimate_time_to_leak(slope: float, score_now: float):
    """Seconds until the trend crosses the anomaly threshold, or None if
    not trending toward it / already crossed long ago / too far out to
    be a meaningful estimate."""
    if score_now >= ANOMALY_THRESHOLD:
        return 0.0  # already anomalous
    if slope <= 0:
        return None  # not trending toward a leak
    seconds = (ANOMALY_THRESHOLD - score_now) / slope
    if seconds > MAX_HORIZON_SECONDS:
        return None
    return seconds


def primary_driver(feat_row: pd.Series, position: str) -> str:
    """Which engineered feature is most deviated right now — a
    human-readable explanation of *why* a region was flagged."""
    candidates = {
        f"moisture_{position}_raw_zscore": "moisture rising",
        f"pressure_{position}_value_zscore": "pressure deviating",
    }
    best_label, best_val = "unknown", 0.0
    for col, label in candidates.items():
        if col in feat_row.index and pd.notna(feat_row[col]):
            val = abs(feat_row[col])
            if val > best_val:
                best_label, best_val = label, val
    return best_label


def format_duration(seconds) -> str:
    if seconds is None:
        return "stable"
    if seconds <= 0:
        return "now"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def run(db_path: Path) -> pd.DataFrame:
    scores = load_table(db_path, "anomaly_scores")
    if scores.empty:
        return pd.DataFrame()
    features = load_table(db_path, "features")

    rows = []
    for (device_id, position), g in scores.groupby(["device_id", "position"]):
        g = g.sort_values("ts").tail(WINDOW_POINTS)
        if len(g) < MIN_POINTS:
            continue

        slope, score_now = fit_trend(g["ts"], g["anomaly_score"])
        eta_seconds = estimate_time_to_leak(slope, score_now)
        latest_ts = g["ts"].iloc[-1]

        feat_row = pd.Series(dtype=float)
        if not features.empty:
            fg = features[(features["device_id"] == device_id) & (features["ts"] <= latest_ts)]
            if not fg.empty:
                feat_row = fg.sort_values("ts").iloc[-1]

        rows.append(
            {
                "device_id": device_id,
                "position": position,
                "ts": latest_ts,
                "current_anomaly_score": score_now,
                "trend_slope_per_min": slope * 60,
                "estimated_seconds_to_leak": eta_seconds,
                "estimated_time_to_leak": format_duration(eta_seconds),
                "primary_driver": primary_driver(feat_row, position) if not feat_row.empty else "unknown",
            }
        )

    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, db_path: Path, csv_path: Path | None = None):
    conn = sqlite3.connect(db_path)
    df.to_sql("time_to_leak_estimates", conn, if_exists="replace", index=False)
    conn.close()

    if csv_path:
        df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to sensor_data.db")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "time_to_leak_estimates.csv", help="CSV output path")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"No database found at {args.db}. Run anomaly_detection.py first.")

    results = run(args.db)
    if results.empty:
        raise SystemExit("Not enough anomaly-score history yet to fit trends (need 5+ points per region).")

    save_results(results, args.db, args.out)

    print(
        results[
            ["device_id", "position", "current_anomaly_score", "trend_slope_per_min", "estimated_time_to_leak", "primary_driver"]
        ].to_string(index=False)
    )
    print(f"\nSaved to table 'time_to_leak_estimates' in {args.db} and CSV at {args.out}")


if __name__ == "__main__":
    main()
