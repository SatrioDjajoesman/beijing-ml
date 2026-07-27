#!/usr/bin/env python3
"""
Live inference service — Stage D of the leak-prediction pipeline.

Connects to esp32/receive_data.py (ws://<host>:8080) as a viewer client,
consumes live sensor updates, and scores them in real time against the
FROZEN anomaly models trained by anomaly_detection.py. Models are only
ever loaded here, never retrained. If this service retrained on live
data the way anomaly_detection.py does for batch analysis, a leak
developing in front of it would keep getting folded into its own
definition of "normal" and the score would never cross the threshold.

Publishes predictions on a second WebSocket server (port 8090) as
{"type": "prediction", "device": ..., "position": ..., ...} for any
dashboard/frontend to subscribe to, and logs every prediction to the
`live_predictions` table in sensor_data.db.

Before running this:
    1. Collect baseline (leak-free) data with esp32/receive_data.py
    2. Run feature_engineering.py against that baseline data
    3. Run anomaly_detection.py against it once to freeze models in ml/models/

Usage:
    ../esp32/venv/bin/python3 live_inference.py [--source ws://localhost:8080] [--db PATH]
"""

import argparse
import asyncio
import json
import math
import sqlite3
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import websockets

from db_path import default_db_path
from feature_engineering import (
    WINDOWS,
    add_pressure_deltas,
    add_rolling_features,
    add_wet_duration,
    pivot_wide,
)
from time_to_leak import ANOMALY_THRESHOLD, estimate_time_to_leak, fit_trend, format_duration, primary_driver

DEFAULT_DB_PATH = default_db_path()
MODELS_DIR = Path(__file__).parent / "models"
POSITIONS = ["start", "middle", "end"]

BUFFER_MAXLEN = 400  # raw readings kept per device (enough for a 10min rolling window at a few-second rate)
SCORE_HISTORY_LEN = 20  # matches time_to_leak.WINDOW_POINTS
MIN_SCORE_POINTS = 5  # matches time_to_leak.MIN_POINTS
DASHBOARD_PORT = 8090


class DeviceState:
    def __init__(self):
        self.raw_rows = deque(maxlen=BUFFER_MAXLEN)
        self.score_history = {p: deque(maxlen=SCORE_HISTORY_LEN) for p in POSITIONS}


device_states = {}
loaded_models = {}
dashboard_clients = set()
latest_predictions = {}  # (device_id, position) -> last prediction payload
db_conn = None


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            device_id TEXT NOT NULL,
            position TEXT NOT NULL,
            anomaly_score REAL,
            is_anomaly INTEGER,
            estimated_time_to_leak TEXT,
            primary_driver TEXT
        )
        """
    )
    conn.commit()
    return conn


def load_model(device_id: str, position: str):
    key = (device_id, position)
    if key in loaded_models:
        return loaded_models[key]

    path = MODELS_DIR / f"isoforest_{device_id}_{position}.joblib"
    if not path.exists():
        loaded_models[key] = None
        return None

    bundle = joblib.load(path)
    loaded_models[key] = bundle
    return bundle


def rows_from_payload(ts: str, device_id: str, data: dict) -> list[dict]:
    rows = []
    for position, sensor in data.get("moisture", {}).items():
        rows.append(
            {
                "ts": ts,
                "device_id": device_id,
                "sensor_type": "moisture",
                "position": position,
                "raw": sensor.get("raw"),
                "status": sensor.get("status"),
                "ready": None,
                "value": None,
            }
        )
    for position, sensor in data.get("pressure", {}).items():
        ready = sensor.get("ready")
        rows.append(
            {
                "ts": ts,
                "device_id": device_id,
                "sensor_type": "pressure",
                "position": position,
                "raw": None,
                "status": None,
                "ready": int(bool(ready)) if ready is not None else None,
                "value": sensor.get("value"),
            }
        )
    return rows


def compute_features(state: DeviceState, device_id: str) -> pd.DataFrame:
    df = pd.DataFrame(state.raw_rows)
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    df["device_id"] = device_id

    wide = pivot_wide(df)
    if wide.empty:
        return pd.DataFrame()

    feats = add_rolling_features(wide)
    feats = add_pressure_deltas(feats)
    feats = add_wet_duration(feats)
    return feats.sort_values("ts")


def safe_num(value):
    """Convert a pandas/numpy scalar to a JSON-safe float, or None if
    missing/NaN (NaN has no JSON representation)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 6)


def signal_trace(feat_row: pd.Series, col: str) -> dict:
    """Every intermediate number feature_engineering.py computed for one
    raw signal column (e.g. "moisture_start_raw"): the reading itself,
    its rolling mean/std at each window, its rate of change, and its
    z-score against its own history."""
    return {
        "column": col,
        "raw": safe_num(feat_row.get(col)),
        "rolling": {
            window: {
                "mean": safe_num(feat_row.get(f"{col}_mean_{window}")),
                "std": safe_num(feat_row.get(f"{col}_std_{window}")),
            }
            for window in WINDOWS
        },
        "rate_of_change": safe_num(feat_row.get(f"{col}_roc")),
        "zscore": safe_num(feat_row.get(f"{col}_zscore")),
    }


def score_current(device_id: str, feat_row: pd.Series, position: str):
    bundle = load_model(device_id, position)
    if bundle is None:
        return None

    model, columns = bundle["model"], bundle["columns"]
    if any(c not in feat_row.index or pd.isna(feat_row[c]) for c in columns):
        return None

    input_vector = [feat_row[c] for c in columns]
    X = [input_vector]
    raw_decision = float(model.decision_function(X)[0])
    score = -raw_decision
    is_anomaly = model.predict(X)[0] == -1
    return {
        "score": score,
        "raw_decision": raw_decision,
        "is_anomaly": is_anomaly,
        "columns": columns,
        "input_vector": input_vector,
        "n_estimators": model.n_estimators,
    }


def broadcast_prediction(payload: dict):
    if dashboard_clients:
        websockets.broadcast(dashboard_clients, json.dumps(payload))


def log_prediction(payload: dict):
    db_conn.execute(
        "INSERT INTO live_predictions (ts, device_id, position, anomaly_score, is_anomaly, estimated_time_to_leak, primary_driver) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            payload["ts"],
            payload["device"],
            payload["position"],
            payload["anomaly_score"],
            int(payload["is_anomaly"]),
            payload["estimated_time_to_leak"],
            payload["primary_driver"],
        ),
    )
    db_conn.commit()


async def process_update(device_id: str, data: dict):
    ts = datetime.now().isoformat()
    state = device_states.setdefault(device_id, DeviceState())
    state.raw_rows.extend(rows_from_payload(ts, device_id, data))

    feats = compute_features(state, device_id)
    if feats.empty:
        return
    feat_row = feats.iloc[-1]

    for position in POSITIONS:
        result = score_current(device_id, feat_row, position)
        if result is None:
            continue
        score = result["score"]
        is_anomaly = result["is_anomaly"]

        hist = state.score_history[position]
        hist.append((feat_row["ts"], score))

        trend_trace = None
        eta_trace = None
        if len(hist) >= MIN_SCORE_POINTS:
            times = pd.Series([h[0] for h in hist])
            scores = pd.Series([h[1] for h in hist])
            slope, score_now = fit_trend(times, scores)
            eta_seconds = estimate_time_to_leak(slope, score_now)
            eta_label = format_duration(eta_seconds)

            trend_trace = {
                "history": [
                    {"t": t.isoformat(), "score": safe_num(s)} for t, s in hist
                ],
                "slope_per_second": safe_num(slope),
                "fitted_score_now": safe_num(score_now),
            }
            eta_trace = {
                "formula": "(threshold - fitted_score_now) / slope_per_second",
                "threshold": ANOMALY_THRESHOLD,
                "fitted_score_now": safe_num(score_now),
                "slope_per_second": safe_num(slope),
                "result_seconds": safe_num(eta_seconds),
                "result_label": eta_label,
            }
        else:
            eta_label = "collecting data"

        moisture_col = f"moisture_{position}_raw"
        pressure_col = f"pressure_{position}_value"
        moisture_z = safe_num(feat_row.get(f"{moisture_col}_zscore"))
        pressure_z = safe_num(feat_row.get(f"{pressure_col}_zscore"))
        driver_label = primary_driver(feat_row, position)

        delta_cols = [c for c in result["columns"] if "minus" in c]
        wet_col = f"moisture_{position}_wet_seconds"

        trace = {
            "position": position,
            "raw_inputs": {
                "moisture": signal_trace(feat_row, moisture_col),
                "pressure": signal_trace(feat_row, pressure_col),
            },
            "cross_position_deltas": {
                c: safe_num(feat_row.get(c)) for c in delta_cols
            },
            "wet_duration_seconds": safe_num(feat_row.get(wet_col)),
            "model_input": {
                "columns": result["columns"],
                "values": [safe_num(v) for v in result["input_vector"]],
                "n_features": len(result["columns"]),
            },
            "model_output": {
                "algorithm": "IsolationForest",
                "n_estimators": result["n_estimators"],
                "raw_decision_function": safe_num(result["raw_decision"]),
                "anomaly_score": safe_num(score),
                "score_formula": "anomaly_score = -decision_function(input_vector)",
                "threshold": ANOMALY_THRESHOLD,
                "is_anomaly": bool(is_anomaly),
            },
            "trend_fit": trend_trace,
            "eta_calculation": eta_trace,
            "primary_driver_calculation": {
                "moisture_zscore_abs": abs(moisture_z) if moisture_z is not None else None,
                "pressure_zscore_abs": abs(pressure_z) if pressure_z is not None else None,
                "winner": driver_label,
            },
        }

        payload = {
            "type": "prediction",
            "ts": ts,
            "device": device_id,
            "position": position,
            "anomaly_score": round(float(score), 4),
            "is_anomaly": bool(is_anomaly),
            "estimated_time_to_leak": eta_label,
            "primary_driver": driver_label,
            "trace": trace,
        }

        print(
            f"[{ts}] {device_id}/{position}: score={payload['anomaly_score']} "
            f"anomaly={payload['is_anomaly']} eta={eta_label} driver={payload['primary_driver']}"
        )

        latest_predictions[(device_id, position)] = payload
        broadcast_prediction(payload)
        log_prediction(payload)


async def consume_source(source_url: str):
    while True:
        try:
            async with websockets.connect(source_url) as ws:
                await ws.send(json.dumps({"type": "hello", "role": "viewer"}))
                print(f"[+] Connected to sensor source at {source_url}")

                async for message in ws:
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("type") == "snapshot":
                        for device_id, data in msg.get("devices", {}).items():
                            await process_update(device_id, data)
                    elif msg.get("type") == "update":
                        await process_update(msg["device"], msg["data"])

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[-] Lost connection to sensor source ({e}); retrying in 5s...")
            await asyncio.sleep(5)


async def handle_dashboard_client(websocket):
    dashboard_clients.add(websocket)
    # Replay whatever we already know so a client that connects mid-session
    # doesn't sit blank until the next scoring cycle happens to fire.
    for payload in latest_predictions.values():
        await websocket.send(json.dumps(payload))
    try:
        await websocket.wait_closed()
    finally:
        dashboard_clients.discard(websocket)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="ws://localhost:8080", help="WebSocket URL of receive_data.py")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    global db_conn
    db_conn = init_db(args.db)

    if not any(MODELS_DIR.glob("isoforest_*.joblib")):
        print(
            f"[!] No frozen models found in {MODELS_DIR}/. Run anomaly_detection.py against a "
            f"baseline dataset first — predictions will be skipped until models exist."
        )

    print(f"Starting dashboard WebSocket server on ws://0.0.0.0:{DASHBOARD_PORT}")
    async with websockets.serve(handle_dashboard_client, "0.0.0.0", DASHBOARD_PORT):
        await consume_source(args.source)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nLive inference stopped.")
