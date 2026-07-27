#!/usr/bin/env python3
"""
Simulated benchmark for the leak-prediction pipeline.

IMPORTANT — what this measures and what it doesn't:
This runs the REAL production code (feature_engineering.py's rolling-stat/
zscore/delta logic, the same IsolationForest config anomaly_detection.py
uses, and time_to_leak.py's trend-fit/ETA formula) against SYNTHETIC sensor
data with a known, programmed leak start time. It answers "how does this
pipeline behave against a leak shaped like our synthetic model" — it is
NOT a real-world validated accuracy number, because no real staged leak
test has been run yet. Treat every number this prints as a benchmark on
simulated data, not a claim about real hardware performance.

Methodology:
  1. Generate a long, leak-free baseline for one synthetic pipe and freeze
     one IsolationForest per position from it (same as anomaly_detection.py
     would from real baseline data).
  2. Run many trials: continue the time series, inject a leak at a known
     position with a known start time and a known ramp speed (slow/medium/
     fast), and replay it sample-by-sample the way live_inference.py would
     see it arrive live.
  3. Record when each position's model first raises a SUSTAINED anomaly
     flag (3 consecutive anomalous samples, to filter single-sample noise),
     and compare that timestamp to the true leak start time.
  4. Also run pure no-leak trials to measure the false-positive rate during
     normal operation.
  5. Aggregate: detection latency, detection rate, localization accuracy,
     false-positive rate, precision/recall/F1, and ETA lead-time accuracy.

Usage:
    ../esp32/venv/bin/python3 evaluate.py
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaly_detection import position_feature_columns
from feature_engineering import add_pressure_deltas, add_rolling_features, add_wet_duration, pivot_wide
from time_to_leak import ANOMALY_THRESHOLD, MIN_POINTS, WINDOW_POINTS, estimate_time_to_leak, fit_trend

POSITIONS = ["start", "middle", "end"]
DT_SECONDS = 5
DEVICE_ID = "eval-pipe"

PRESSURE_BASE = {"start": 100.0, "middle": 99.0, "end": 98.0}
MOISTURE_BASE = 480.0
MOISTURE_NOISE = 5.0
PRESSURE_NOISE = 0.3

RAMP_RATES = {
    # (moisture units per sample, pressure-drop units per sample)
    "slow": (1.0, 0.02),
    "medium": (4.0, 0.08),
    "fast": (12.0, 0.25),
}

BASELINE_SAMPLES = 3000  # ~4h at 5s cadence
WARMUP_SAMPLES = 250  # normal samples used only to mature rolling windows before the analyzed period starts (discarded from all metrics)
LEAD_IN_SAMPLES = 200  # normal samples immediately before a leak trial, to prime rolling windows
TRIAL_SAMPLES_AFTER_LEAK = 300  # ~25min at 5s cadence
CONFIRM_STREAK = 3  # consecutive anomalous samples required to call it "detected", not noise
TRIALS_PER_COMBINATION = 5
NO_LEAK_TRIALS = 10

CONFIDENCE_ADJACENT = {"start": ["middle"], "middle": ["start", "end"], "end": ["middle"]}


def make_rows(ts: datetime, values: dict) -> list[dict]:
    rows = []
    for position in POSITIONS:
        rows.append(
            {
                "ts": ts, "device_id": DEVICE_ID, "sensor_type": "moisture", "position": position,
                "raw": values[position]["moisture"], "status": "wet" if values[position]["moisture"] > 700 else "dry",
                "ready": None, "value": None,
            }
        )
        rows.append(
            {
                "ts": ts, "device_id": DEVICE_ID, "sensor_type": "pressure", "position": position,
                "raw": None, "status": None, "ready": 1, "value": values[position]["pressure"],
            }
        )
    return rows


def generate_series(rng: random.Random, start_time: datetime, n: int, leak_position=None, leak_start_index=None, ramp=None) -> list[dict]:
    rows = []
    for i in range(n):
        ts = start_time + timedelta(seconds=DT_SECONDS * i)
        values = {}
        for position in POSITIONS:
            moisture = MOISTURE_BASE + rng.uniform(-MOISTURE_NOISE, MOISTURE_NOISE)
            pressure = PRESSURE_BASE[position] + rng.uniform(-PRESSURE_NOISE, PRESSURE_NOISE)
            values[position] = {"moisture": moisture, "pressure": pressure}

        if leak_position is not None and leak_start_index is not None and i >= leak_start_index:
            progress = i - leak_start_index
            moist_rate, press_rate = ramp
            values[leak_position]["moisture"] += progress * moist_rate
            values[leak_position]["pressure"] = max(values[leak_position]["pressure"] - progress * press_rate, PRESSURE_BASE[leak_position] * 0.3)
            for neighbor in CONFIDENCE_ADJACENT[leak_position]:
                values[neighbor]["pressure"] = max(values[neighbor]["pressure"] - progress * press_rate * 0.5, PRESSURE_BASE[neighbor] * 0.3)

        rows.extend(make_rows(ts, values))
    return rows


def build_features(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    wide = pivot_wide(df)
    feats = add_rolling_features(wide)
    feats = add_pressure_deltas(feats)
    feats = add_wet_duration(feats)
    return feats.sort_values("ts").reset_index(drop=True)


def train_baseline_models(rng: random.Random):
    start_time = datetime(2026, 1, 1)
    rows = generate_series(rng, start_time, BASELINE_SAMPLES)
    feats = build_features(rows)

    models = {}
    for position in POSITIONS:
        cols = position_feature_columns(feats, position)
        sub = feats[cols].dropna()
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        model.fit(sub.values)
        models[position] = (model, cols)
    return models


def score_position_batch(feats: pd.DataFrame, position: str, model, cols: list) -> pd.DataFrame:
    """Score every row for one position's model in a single batched call,
    instead of one sklearn call per row (which is orders of magnitude
    slower for a series of any real length)."""
    sub = feats[["ts"] + cols].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["ts", "score", "is_anomaly"])

    X = sub[cols].values
    scores = -model.decision_function(X)
    is_anomaly = model.predict(X) == -1
    return pd.DataFrame({"ts": sub["ts"].values, "score": scores, "is_anomaly": is_anomaly})


def run_trial(rng: random.Random, models: dict, leak_position=None, ramp_name=None):
    start_time = datetime(2026, 2, 1)
    n_total = LEAD_IN_SAMPLES + (TRIAL_SAMPLES_AFTER_LEAK if leak_position else BASELINE_SAMPLES // 10)
    leak_start_index = LEAD_IN_SAMPLES if leak_position else None
    ramp = RAMP_RATES[ramp_name] if ramp_name else None

    # Prepend a normal-data warm-up period so the rolling/expanding features
    # are already mature by the time the analyzed period starts, matching
    # how a real rolling buffer behaves after running for a while — a fresh,
    # isolated short series would make rolling stats mismatch what the model
    # was trained on and swamp everything in false positives.
    warmup_start = start_time - timedelta(seconds=DT_SECONDS * WARMUP_SAMPLES)
    warmup_rows = generate_series(rng, warmup_start, WARMUP_SAMPLES)
    trial_rows = generate_series(rng, start_time, n_total, leak_position, leak_start_index, ramp)
    feats_full = build_features(warmup_rows + trial_rows)
    feats = feats_full[feats_full["ts"] >= start_time].reset_index(drop=True)

    leak_start_ts = start_time + timedelta(seconds=DT_SECONDS * leak_start_index) if leak_position else None

    anomaly_flags = {}  # position -> list of (ts, is_anomaly)
    eta_at = {}  # position -> list of (ts, eta_seconds_or_None)

    for position in POSITIONS:
        model, cols = models[position]
        scored = score_position_batch(feats, position, model, cols)

        anomaly_flags[position] = list(zip(scored["ts"], scored["is_anomaly"]))

        # The trend fit needs a sliding window over the score history, which
        # is inherently sequential — but it's cheap (numpy polyfit on <=20
        # points), so doing this in a plain Python loop over one position's
        # ~500 rows is fine; it's the model scoring above that had to be
        # batched.
        eta_list = []
        for i in range(len(scored)):
            window = scored.iloc[max(0, i - WINDOW_POINTS + 1) : i + 1]
            eta_seconds = None
            if len(window) >= MIN_POINTS:
                slope, score_now = fit_trend(window["ts"], window["score"])
                eta_seconds = estimate_time_to_leak(slope, score_now)
            eta_list.append((scored["ts"].iloc[i], eta_seconds))
        eta_at[position] = eta_list

    # Confirmed detection: first ts where CONFIRM_STREAK consecutive samples are anomalous.
    confirmed_detection = {}
    for position in POSITIONS:
        flags = anomaly_flags[position]
        confirmed_detection[position] = None
        for i in range(len(flags) - CONFIRM_STREAK + 1):
            if all(flags[i + k][1] for k in range(CONFIRM_STREAK)):
                confirmed_detection[position] = flags[i][0]
                break

    return {
        "leak_position": leak_position,
        "ramp_name": ramp_name,
        "leak_start_ts": leak_start_ts,
        "confirmed_detection": confirmed_detection,
        "anomaly_flags": anomaly_flags,
        "eta_at": eta_at,
    }


def main():
    master_rng = random.Random(42)
    print("Training baseline models on", BASELINE_SAMPLES, "leak-free samples per position...")
    models = train_baseline_models(master_rng)
    print("Done.\n")

    leak_trials = []
    for leak_position in POSITIONS:
        for ramp_name in RAMP_RATES:
            for trial_i in range(TRIALS_PER_COMBINATION):
                rng = random.Random(hash((leak_position, ramp_name, trial_i)) & 0xFFFFFFFF)
                leak_trials.append(run_trial(rng, models, leak_position, ramp_name))

    no_leak_trials = [run_trial(random.Random(1000 + i), models) for i in range(NO_LEAK_TRIALS)]

    # --- Detection latency & detection rate, grouped by ramp speed ---
    print("=" * 70)
    print("DETECTION LATENCY (simulated leak scenarios)")
    print("=" * 70)
    # Split into genuine post-leak detections (latency >= 0) vs. trials where
    # a false alarm on the correct region fired during normal operation
    # *before* the leak even started — averaging those two cases together
    # would hide the false-positive problem inside a misleading "latency".
    true_latency_by_ramp = {name: [] for name in RAMP_RATES}
    false_alarm_before_leak_by_ramp = {name: 0 for name in RAMP_RATES}
    missed_by_ramp = {name: 0 for name in RAMP_RATES}
    localization_correct = 0
    localization_total = 0

    for trial in leak_trials:
        pos = trial["leak_position"]
        ramp = trial["ramp_name"]
        detected_ts = trial["confirmed_detection"][pos]
        localization_total += 1
        if detected_ts is None:
            missed_by_ramp[ramp] += 1
            continue
        latency = (detected_ts - trial["leak_start_ts"]).total_seconds()
        if latency < 0:
            false_alarm_before_leak_by_ramp[ramp] += 1
        else:
            true_latency_by_ramp[ramp].append(latency)

        # Localization: did the true position get flagged at or before any other position?
        other_detections = [
            trial["confirmed_detection"][p] for p in POSITIONS if p != pos and trial["confirmed_detection"][p] is not None
        ]
        if not other_detections or detected_ts <= min(other_detections):
            localization_correct += 1

    for ramp_name in RAMP_RATES:
        lat = true_latency_by_ramp[ramp_name]
        false_before = false_alarm_before_leak_by_ramp[ramp_name]
        missed = missed_by_ramp[ramp_name]
        total = TRIALS_PER_COMBINATION * len(POSITIONS)
        if lat:
            print(
                f"  {ramp_name:>6} leak: genuine post-leak detection in {len(lat)}/{total} trials | "
                f"latency mean={np.mean(lat):.0f}s median={np.median(lat):.0f}s "
                f"min={np.min(lat):.0f}s max={np.max(lat):.0f}s"
            )
            print(f"           false alarm fired before leak even started in {false_before}/{total} trials | missed={missed}")
        else:
            print(f"  {ramp_name:>6} leak: detected 0/{total} trials | missed={missed}")

    print(f"\nLocalization accuracy (correct region detected first or only): "
          f"{localization_correct}/{localization_total} ({100*localization_correct/localization_total:.1f}%)")

    # --- False positive rate during normal operation ---
    print("\n" + "=" * 70)
    print("FALSE POSITIVE RATE (no-leak trials)")
    print("=" * 70)
    total_samples = 0
    false_positive_samples = 0
    for trial in no_leak_trials:
        for position in POSITIONS:
            flags = [f for _, f in trial["anomaly_flags"][position]]
            total_samples += len(flags)
            false_positive_samples += sum(flags)
    fpr = false_positive_samples / total_samples if total_samples else 0
    hours_per_trial = (LEAD_IN_SAMPLES + BASELINE_SAMPLES // 10) * DT_SECONDS / 3600
    total_hours = hours_per_trial * NO_LEAK_TRIALS * len(POSITIONS)
    false_alarms_per_hour = false_positive_samples / total_hours if total_hours else 0
    print(f"  Per-sample false positive rate: {fpr*100:.2f}% ({false_positive_samples}/{total_samples} samples)")
    print(f"  False alarms per hour of normal operation (per region): {false_alarms_per_hour:.2f}")

    # --- Precision / Recall / F1 over pooled leak + no-leak samples ---
    print("\n" + "=" * 70)
    print("PRECISION / RECALL / F1 (pooled across all trials)")
    print("=" * 70)
    tp = fp = fn = tn = 0
    for trial in leak_trials + no_leak_trials:
        leak_pos = trial["leak_position"]
        leak_start = trial["leak_start_ts"]
        for position in POSITIONS:
            for ts, flag in trial["anomaly_flags"][position]:
                ground_truth = bool(leak_pos and position == leak_pos and leak_start and ts >= leak_start)
                if ground_truth and flag:
                    tp += 1
                elif ground_truth and not flag:
                    fn += 1
                elif not ground_truth and flag:
                    fp += 1
                else:
                    tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}   Accuracy: {accuracy:.3f}")
    print(f"  (TP={tp} FP={fp} FN={fn} TN={tn}, per-sample, per-region, pooled across all trials)")

    # --- ETA lead-time accuracy ---
    print("\n" + "=" * 70)
    print("ETA / TIME-TO-LEAK LEAD-TIME ACCURACY")
    print("=" * 70)
    lead_errors = []
    genuine_lead_count = 0
    for trial in leak_trials:
        pos = trial["leak_position"]
        detected_ts = trial["confirmed_detection"][pos]
        if detected_ts is None:
            continue
        for ts, eta_seconds in trial["eta_at"][pos]:
            if ts >= detected_ts or eta_seconds is None:
                continue
            actual_remaining = (detected_ts - ts).total_seconds()
            if actual_remaining <= 0:
                continue
            genuine_lead_count += 1
            lead_errors.append(abs(eta_seconds - actual_remaining))

    if lead_errors:
        print(f"  Samples with a genuine (non-'now') lead-time estimate before detection: {genuine_lead_count}")
        print(f"  Mean absolute error vs. actual remaining time: {np.mean(lead_errors):.0f}s "
              f"(median {np.median(lead_errors):.0f}s)")
    else:
        print("  No genuine lead-time estimates were given before detection in any trial —")
        print("  the score crossed the anomaly threshold too abruptly for the trend fit to")
        print("  give advance warning in these simulated scenarios (matches earlier findings).")

    print("\n" + "=" * 70)
    print("CAVEAT: all numbers above are from SYNTHETIC data with a programmed leak")
    print("shape, not real hardware. Use these as a benchmark of the pipeline's logic,")
    print("not as a claim about real-world accuracy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
