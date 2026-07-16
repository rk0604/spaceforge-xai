#!/usr/bin/env python3

import csv
import shlex
import sys
from pathlib import Path


def normalize_header(value):
    """
    Normalize CSV headers so small spacing differences do not break parsing.
    """
    if value is None:
        return ""

    cleaned = value.replace("\ufeff", "")
    cleaned = cleaned.strip()
    cleaned = " ".join(cleaned.split())

    return cleaned


def normalize_config_name(value):
    """
    Preserve the visible config name while removing accidental outer spacing.
    """
    if value is None:
        return ""

    return value.strip()


def get_value(row, normalized_header_map, target_header):
    """
    Read a value from a row using a normalized CSV header name.
    """
    normalized_target = normalize_header(target_header)
    actual_header = normalized_header_map.get(normalized_target)

    if actual_header is None:
        return ""

    value = row.get(actual_header, "")

    if value is None:
        return ""

    return str(value).strip()


def require_numeric(value, config_name, csv_header):
    """
    Validate that a CSV value can be passed as a numeric command line argument.
    """
    if value == "":
        raise SystemExit(
            f"missing value for config '{config_name}' in column '{csv_header}'"
        )

    try:
        float(value)
    except ValueError:
        raise SystemExit(
            f"non numeric value for config '{config_name}' in column '{csv_header}': {value}"
        )

    return value


def find_config_row(rows, config_name):
    """
    Find the exact config row requested by the Slurm script.
    """
    for row in rows:
        current_name = normalize_config_name(row.get("Config", ""))

        if current_name == config_name:
            return row

    return None


def find_default_row(rows):
    """
    Find the Config 1 fallback row used when a specific value is blank.
    """
    for row in rows:
        current_name = normalize_config_name(row.get("Config", ""))

        if current_name == "Config 1":
            return row

    raise SystemExit("fallback row 'Config 1' was not found in the Config column")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: config_row_to_args.py <analytics_csv> <config_name>"
        )

    csv_path = Path(sys.argv[1])
    target_config = normalize_config_name(sys.argv[2])

    if not csv_path.exists():
        raise SystemExit(f"tracker CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise SystemExit("tracker CSV has no data rows")

    normalized_header_map = {
        normalize_header(header): header
        for header in fieldnames
    }

    if "Config" not in fieldnames:
        matching_config_header = normalized_header_map.get("Config")
        if matching_config_header is None:
            raise SystemExit("tracker CSV must contain a Config column")

        for row in rows:
            row["Config"] = row.get(matching_config_header, "")

    selected_row = find_config_row(rows, target_config)

    if selected_row is None:
        available = [
            normalize_config_name(row.get("Config", ""))
            for row in rows
            if normalize_config_name(row.get("Config", ""))
        ]

        raise SystemExit(
            "config not found: "
            + target_config
            + "\navailable configs:\n  "
            + "\n  ".join(available)
        )

    default_row = find_default_row(rows)

    column_map = [
        (
            "--battery-capacity-wh",
            "Battery capacity_wh (battery.hpp line 9)",
        ),
        (
            "--battery-start-charge-wh",
            "battery start capacity: charge_ (can't be set, is by default half of capacity)",
        ),
        (
            "--battery-max-discharge-w",
            "battery_max_discharge_W (battery.hpp line 40)",
        ),
        (
            "--battery-max-charge-w",
            "battery_max_charge_W (battery.hpp line 40)",
        ),
        (
            "--effusion-h-wk",
            "h_WK (effusion Cell) (too complicated leave it for Rishab)",
        ),
        (
            "--effusion-c-j",
            "C_J (effusion cell) (too complicated leave it for Rishab)",
        ),
        (
            "--solar-base-input-w",
            "base_input (solar) (main.cpp line 633)",
        ),
        (
            "--solar-efficiency",
            "efficiency (solar) (main.cpp line 632)",
        ),
        (
            "--substrate-c-j",
            "C_J (susbtrateHeater.hpp line 413)",
        ),
        (
            "--substrate-eps",
            "eps (substrateHeater.hpp line 411)",
        ),
        (
            "--substrate-fail-limit-ticks",
            "FAIL_LIMIT_TICKS_ Subsrate (subsrateHeater.hpp line 451)",
        ),
        (
            "--substrate-ready-band-k",
            "READY_BAND_K_ (subsrateHeater.hpp line 450)",
        ),
        (
            "--substrate-max-power-w",
            "Substrate maxPowerDraw Main.cpp line 643",
        ),
        (
            "--effusion-underflux-limit-ticks",
            "Effusion Underflux streak cap (line 870 main.cpp)",
        ),
        (
            "--effusion-undertemp-limit-ticks",
            "Effusion undertemp streak cap (line 870 main.cpp)",
        ),
        (
            "--effusion-min-flux-fraction",
            "Effusion MIN_FLUX_FRACTION (line 870 main.cpp)",
        ),
        (
            "--effusion-temp-tolerance-fraction",
            "Effusion TEMP_TOLERANCE_FRACTION (line 870 main.cpp)",
        ),
        (
            "--heater-bank-max-draw-w",
            "max draw (heater bank) main.cpp line 640",
        ),
    ]

    args = [
        "--config-name",
        target_config,
    ]

    for cli_flag, csv_header in column_map:
        value = get_value(selected_row, normalized_header_map, csv_header)

        if value == "":
            value = get_value(default_row, normalized_header_map, csv_header)

        value = require_numeric(value, target_config, csv_header)

        args.append(cli_flag)
        args.append(value)

    args.extend(
        [
            "--effusion-night-ambient-k",
            "250.0",
            "--effusion-day-ambient-k",
            "325.0",
        ]
    )

    print(" ".join(shlex.quote(item) for item in args))


if __name__ == "__main__":
    main()