"""
Resolves the default sensor_data.db location without hardcoding the ESP32
backend folder's name. Clones of this repo have renamed that folder (e.g.
esp32/, final-esp32-backend/, or a friend's own name for it), which broke
every script here defaulting to a literal "esp32/sensor_data.db" path.

Finds sensor_data.db in any sibling of ml/, preferring the most recently
written one if more than one exists. Falls back to the historical
esp32/sensor_data.db path if none exists yet, so a fresh clone still gets
a sensible location to create the db at.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def default_db_path() -> Path:
    candidates = sorted(
        PROJECT_ROOT.glob("*/sensor_data.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return PROJECT_ROOT / "esp32" / "sensor_data.db"
