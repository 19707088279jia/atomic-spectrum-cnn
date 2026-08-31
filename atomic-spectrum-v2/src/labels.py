"""Six-target labels and frozen NASA Z-903 thresholds."""

from __future__ import annotations

TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
CNN_THRESHOLDS = {
    "Zn": 0.7050,
    "Mn": 0.2350,
    "Cd": 0.5000,
    "Mg": 0.3550,
    "Cu": 0.5050,
    "Pb": 0.4050,
}
RAW_THRESHOLDS = {
    "Zn": 100.0,
    "Mn": 500.0,
    "Cd": 0.05,
    "Mg": 5.0,
    "Cu": 25.0,
    "Pb": 10.0,
}
UNITS = {"Zn": "ppm", "Mn": "ppm", "Cd": "ppm", "Mg": "oxide wt%", "Cu": "ppm", "Pb": "ppm"}
METADATA_COLUMNS = {"Zn": "Zn", "Mn": "Mn", "Cd": "Cd", "Mg": "MgO", "Cu": "Cu", "Pb": "Pb"}


def composition_label(element: str, raw_value: float | None) -> str:
    """Convert known raw composition to a display label."""
    if raw_value is None:
        return "UNKNOWN"
    return "POSITIVE" if raw_value > RAW_THRESHOLDS[element] else "NEGATIVE"


__all__ = ["TARGETS", "CNN_THRESHOLDS", "RAW_THRESHOLDS", "UNITS", "METADATA_COLUMNS", "composition_label"]
