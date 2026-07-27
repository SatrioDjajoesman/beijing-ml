#!/usr/bin/env python3
"""
Builds ML-ready features from the raw sensor log in esp32/sensor_data.db.

Turns the long-format `sensor_readings` table (one row per sensor per
message) into a wide, per-timestamp feature table per pipe (device_id):
rolling statistics, rate of change, cross-position pressure deltas, and
moisture "wet duration" — the inputs the leak-prediction models will
consume.

Reuses esp32/venv (pandas/numpy installed there).

Usage:
    ../esp32/venv/bin/python3 feature_engineering.py [--db PATH] [--out features.csv]
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).parent.parent / "esp32" / "sensor_data.db"

# Rolling window sizes, expressed as pandas time offsets.
WINDOWS = ["30s", "2min", "10min"]

POSITIONS_ORDER = ["start", "middle", "end"]


def load_readings(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM sensor_readings", conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse long-format readings to one row per (device_id, ts)."""
    frames = []

    moisture = df[df["sensor_type"] == "moisture"]
    if not moisture.empty:
        m_raw = moisture.pivot_table(index=["device_id", "ts"], columns="position", values="raw", aggfunc="last")
        m_raw.columns = [f"moisture_{c}_raw" for c in m_raw.columns]
        frames.append(m_raw)

        m_status = moisture.pivot_table(index=["device_id", "ts"], columns="position", values="status", aggfunc="last")
        m_status.columns = [f"moisture_{c}_status" for c in m_status.columns]
        frames.append(m_status)

    pressure = df[df["sensor_type"] == "pressure"]
    if not pressure.empty:
        p_val = pressure.pivot_table(index=["device_id", "ts"], columns="position", values="value", aggfunc="last")
        p_val.columns = [f"pressure_{c}_value" for c in p_val.columns]
        frames.append(p_val)

    if not frames:
        return pd.DataFrame()

    wide = pd.concat(frames, axis=1).sort_index()
    return wide.reset_index()


def add_rolling_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Per device: rolling mean/std, rate of change, expanding z-score."""
    signal_cols = [c for c in wide.columns if c.endswith("_raw") or c.endswith("_value")]

    out_frames = []
    for _device_id, g in wide.groupby("device_id"):
        g = g.sort_values("ts").set_index("ts")
        feats = g.copy()

        for col in signal_cols:
            series = g[col]

            for window in WINDOWS:
                feats[f"{col}_mean_{window}"] = series.rolling(window).mean()
                feats[f"{col}_std_{window}"] = series.rolling(window).std()

            dt_seconds = series.index.to_series().diff().dt.total_seconds()
            feats[f"{col}_roc"] = series.diff() / dt_seconds

            # Expanding baseline so early history never looks ahead at
            # readings that haven't happened yet.
            exp_mean = series.expanding(min_periods=5).mean()
            exp_std = series.expanding(min_periods=5).std().replace(0, np.nan)
            feats[f"{col}_zscore"] = (series - exp_mean) / exp_std

        out_frames.append(feats.reset_index())

    return pd.concat(out_frames, ignore_index=True)


def add_pressure_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-position pressure gradients, used to localize where along
    the pipe a flow loss is happening."""
    cols = {p: f"pressure_{p}_value" for p in POSITIONS_ORDER}
    if all(c in df.columns for c in cols.values()):
        df["pressure_start_minus_middle"] = df[cols["start"]] - df[cols["middle"]]
        df["pressure_middle_minus_end"] = df[cols["middle"]] - df[cols["end"]]
        df["pressure_start_minus_end"] = df[cols["start"]] - df[cols["end"]]
    return df


def add_wet_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds since each moisture sensor's current wet streak began
    (0 while dry) — a sustained wet reading is a stronger leak signal
    than a brief blip."""
    for pos in POSITIONS_ORDER:
        status_col = f"moisture_{pos}_status"
        if status_col not in df.columns:
            continue

        out_col = f"moisture_{pos}_wet_seconds"
        result = pd.Series(index=df.index, dtype=float)

        for _device_id, g in df.groupby("device_id"):
            g = g.sort_values("ts")
            is_wet = g[status_col].fillna("").str.lower().eq("wet")
            streak_start = g["ts"].where(is_wet & ~is_wet.shift(fill_value=False)).ffill()
            wet_seconds = (g["ts"] - streak_start).dt.total_seconds().where(is_wet, 0.0)
            result.loc[g.index] = wet_seconds

        df[out_col] = result

    return df


def build_feature_table(db_path: Path) -> pd.DataFrame:
    raw = load_readings(db_path)
    if raw.empty:
        return pd.DataFrame()

    wide = pivot_wide(raw)
    if wide.empty:
        return pd.DataFrame()

    feats = add_rolling_features(wide)
    feats = add_pressure_deltas(feats)
    feats = add_wet_duration(feats)
    return feats.sort_values(["device_id", "ts"]).reset_index(drop=True)


def save_features(df: pd.DataFrame, db_path: Path, csv_path: Path | None = None):
    conn = sqlite3.connect(db_path)
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()

    if csv_path:
        df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to sensor_data.db")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "features.csv", help="CSV output path")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"No database found at {args.db}. Run esp32/receive_data.py first to collect data.")

    feats = build_feature_table(args.db)
    if feats.empty:
        raise SystemExit("No sensor readings found in the database yet.")

    save_features(feats, args.db, args.out)
    print(f"Built {len(feats)} feature rows across {feats['device_id'].nunique()} device(s).")
    print(f"Saved to table 'features' in {args.db} and CSV at {args.out}")


if __name__ == "__main__":
    main()
