import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

from .reasoning_processes import get_reasoning_process_for_fault_category
from .schema_columns import INCIDENT_ID_COLUMN, REQUIRED_COLUMNS

# For filter by problem code: synthetic uses fault_category (workflow IDs). Keep in sync with filter_rows.py.
ALLOWED_PROBLEM_CODES = [
    "Service-off",
    "Degraded Prach",
    "Offline / Unreachable",
    "Disabled Cells",
    "Node Down",
    "power_ac_failure_recovery",
    "power_dc_rectifier_recovery",
    "power_battery_discharge_response",
    "power_generator_failure_recovery",
    "env_high_temperature_response",
    "env_hvac_fault_recovery",
    "env_water_intrusion_response",
    "ran_software_upgrade_recovery",
    "compute_vm_failure_recovery",
    "ran_cell_site_down_recovery",
    "compute_container_crash_recovery",
    "ran_interference_mitigation",
    "compute_orchestrator_recovery",
    "ran_speed_complaint_resolution",
    "ran_voice_quality_resolution",
    "compute_image_pull_recovery",
    "ran_sector_outage_recovery",
    "transport_routing_flap_resolution",
    "transport_microwave_degradation_response",
    "signaling_routing_failure_recovery",
    "compute_k8s_node_recovery",
    "ran_prb_availability_resolution",
    "compute_storage_failure_recovery",
    "ran_cell_overshooting_correction",
    "ran_rru_communication_recovery",
    "env_battery_temperature_response",
    "compute_cnf_pod_recovery",
    "signaling_delay_resolution",
    "ran_dropped_calls_resolution",
    "ran_parameter_correction",
    "ran_antenna_tilt_recovery",
    "ran_vswr_alarm_resolution",
    "ran_handover_failure_resolution",
]


def coerce_to_str(value) -> str:
    if pd.isna(value):
        return ""
    # Ensure any non-string is converted to string safely
    try:
        return str(value)
    except Exception:
        return ""


def extract_examples_by_problem_code(df, num):
    fc_col = "fault_category" if "fault_category" in df.columns else "u_problem_code"
    filtered_df = (
        df[df[fc_col].isin(ALLOWED_PROBLEM_CODES)]
        .groupby(fc_col, group_keys=False)
        .apply(lambda x: x.sample(n=min(num, len(x)), random_state=42))
        .reset_index(drop=True)
    )
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description="Convert incidents CSV to input.jsonl for Qwen prompt")
    parser.add_argument("--input", required=True, help="Path to anonymized-Incidents_Last_6_Months.csv")
    parser.add_argument("--output", default="input.jsonl", help="Path to output JSONL (default: input.jsonl)")
    parser.add_argument("--limit", required=False, help="Max number of rows to export")
    parser.add_argument("--required_incidents", required=False, type=str, help="Path to required incidents file")
    parser.add_argument("--examples_by_problem_code", required=False, type=int)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input CSV not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Read CSV (handle BOM if present)
    try:
        df = pd.read_csv(input_path, dtype=str, encoding="latin1")
    except UnicodeDecodeError:
        # Fallback to default encoding
        df = pd.read_csv(input_path, dtype=str)

    if args.examples_by_problem_code:
        df = extract_examples_by_problem_code(df, args.examples_by_problem_code)
        print(f"Extracted {len(df)} examples from the dataset.")

    # Reasoning process from fault_category (synthetic) or u_problem_code (legacy)
    fc_col = "fault_category" if "fault_category" in df.columns else "u_problem_code"
    df["problem_code_reasoning_process"] = df[fc_col].map(get_reasoning_process_for_fault_category)

    # time_to_resolve: synthetic uses intervention_began / intervention_completed
    start_col = "intervention_began" if "intervention_began" in df.columns else "work_start"
    end_col = "intervention_completed" if "intervention_completed" in df.columns else "work_end"
    try:
        df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
        df["time_to_resolve"] = (df[end_col] - df[start_col]).dt.total_seconds().fillna(0).astype(int)
    except Exception as e:
        print(f"Failed to parse datetime columns {start_col}, {end_col}: {e}")
        df["time_to_resolve"] = 0

    original_cols_lower = {c.lower(): c for c in df.columns}

    # Ensure all required columns exist (case-insensitive match). If missing, create empty.
    missing: List[str] = []
    selected = {}
    for col in REQUIRED_COLUMNS:
        # Match case-insensitively
        actual = original_cols_lower.get(col.lower())
        if actual is None:
            # Create empty column if missing
            missing.append(col)
            df[col] = ""
            selected[col] = col
        else:
            selected[col] = actual

    if missing:
        print(
            "WARNING: Missing columns in CSV; blank fields will be used: " + ", ".join(missing),
            file=sys.stderr,
        )

    if args.required_incidents:
        with open(args.required_incidents, "r") as f:
            required_incidents = f.read().splitlines()
        print(required_incidents)
        id_col = INCIDENT_ID_COLUMN if INCIDENT_ID_COLUMN in df.columns else "number"
        df = df[df[id_col].astype(str).isin(required_incidents)]

    # Limit rows
    if args.limit is not None and int(args.limit) > 0:
        df = df.head(int(args.limit))

    # Prepare records
    records = []
    for _, row in df.iterrows():
        rec = {key: coerce_to_str(row[selected[key]]) for key in REQUIRED_COLUMNS}
        records.append(rec)

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_path}.")


if __name__ == "__main__":
    main()
