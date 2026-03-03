# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import pandas as pd

# Legacy u_problem_code values (original pipeline)
ALLOWED_PROBLEM_CODES_LEGACY = [
    "Service-off",
    "Degraded Prach",
    "Offline / Unreachable",
    "Disabled Cells",
    "Node Down",
    "Site Not Scrolling",
    "Sleepy Cell",
    "VM is in not ready state",
    "Prach 0",
    "N2 Link Down",
    "ueconmgr pod restarted",
    "CSR Not Reachable",
    "Circuit Down",
    "Link Down",
    "GPS Sync",
    "MTA Alert",
]

# Synthetic fault_category (workflow IDs from telco_synthetic_from_scratch).
ALLOWED_PROBLEM_CODES_SYNTHETIC = [
    # Power / Environment
    "power_ac_failure_recovery",
    "power_dc_rectifier_recovery",
    "power_battery_discharge_response",
    "power_generator_failure_recovery",
    "env_high_temperature_response",
    "env_hvac_fault_recovery",
    "env_water_intrusion_response",
    "env_battery_temperature_response",
    "env_cabinet_intrusion_response",
    # RAN
    "ran_software_upgrade_recovery",
    "ran_cell_site_down_recovery",
    "ran_interference_mitigation",
    "ran_speed_complaint_resolution",
    "ran_voice_quality_resolution",
    "ran_sector_outage_recovery",
    "ran_prb_availability_resolution",
    "ran_cell_overshooting_correction",
    "ran_rru_communication_recovery",
    "ran_dropped_calls_resolution",
    "ran_parameter_correction",
    "ran_antenna_tilt_recovery",
    "ran_vswr_alarm_resolution",
    "ran_handover_failure_resolution",
    "ran_backhaul_degradation_resolution",
    "ran_cell_congestion_management",
    "ran_device_issue_resolution",
    # Compute
    "compute_vm_failure_recovery",
    "compute_container_crash_recovery",
    "compute_orchestrator_recovery",
    "compute_image_pull_recovery",
    "compute_k8s_node_recovery",
    "compute_storage_failure_recovery",
    "compute_cnf_pod_recovery",
    "compute_resource_exhaustion_resolution",
    # Transport
    "transport_routing_flap_resolution",
    "transport_microwave_degradation_response",
    "transport_interface_errors_resolution",
    "transport_packet_loss_resolution",
    # Signaling
    "signaling_routing_failure_recovery",
    "signaling_delay_resolution",
    "signaling_s1_n2_recovery",
    "signaling_sip_registration_recovery",
]

# Combined: use for filtering so both legacy and synthetic data are supported.
ALLOWED_PROBLEM_CODES = ALLOWED_PROBLEM_CODES_LEGACY + ALLOWED_PROBLEM_CODES_SYNTHETIC


ALLOWED_CLOSE_CODES = [
    "Commercial Power Restored",
    "Power Restored",
    "Network Fix",
    "Cleared In Testing",
    "Solved Remotely (Permanently)",
    "Reset RU",
    "Fiber Repaired/Replaced",
    "Cold Reboot",
    "Performance Improvement",
    "Configuration corrected",
    "Software Fix",
    "Delete M-Plane and F1C IP",
    "RU Reset",
    "Other",
    "Restart MVRP Services",
    "Activity Completed",
    # Synthetic resolution_method values (telco_synthetic_from_scratch)
    "Resolved",
    "Issue Corrected",
    "Service Restored",
    "Pending Resolution",
    "Partial Resolution",
]


def _load_csv(path: str) -> pd.DataFrame:
    """Load CSV, falling back from latin1 to default encoding."""
    try:
        return pd.read_csv(path, encoding="latin1")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def filter_auto(input_csv, output_csv):
    """Filter incident data and save results to a new CSV."""
    df = _load_csv(input_csv)

    # Synthetic schema: resolution_method, resolution_summary
    res_col = df["resolution_method"] if "resolution_method" in df.columns else df["close_code"]
    notes_col = df["resolution_summary"] if "resolution_summary" in df.columns else df["close_notes"]
    mask_auto_recovered = res_col.astype(str).str.contains("Auto Recover", case=False, na=False)
    mask_event_cleared = notes_col.astype(str).str.contains("No Action Taken", case=False, na=False)
    mask_event_cleared_ar = notes_col.astype(str).str.contains("auto recovered", case=False, na=False)
    mask_event_cleared_ar_d = notes_col.astype(str).str.contains("auto-recovered", case=False, na=False)
    mask_remove = mask_auto_recovered | mask_event_cleared | mask_event_cleared_ar | mask_event_cleared_ar_d

    filtered_df = df[~mask_remove]
    filtered_df.to_csv(output_csv, index=False)
    print(f"Original rows: {len(df)}")
    print(f"New rows: {len(filtered_df)}")
    print(f"Auto Rows removed: {mask_remove.sum()}")


def filter_soft_solve(input_csv, output_csv):
    """Filter incident data to keep only soft_solve rows."""
    df = _load_csv(input_csv)

    soft_solve_rows = df[df["solved_category"] == "soft_solve"]

    soft_solve_rows.to_csv(output_csv, index=False)
    print(f"Original rows: {len(df)}")
    print(f"New rows: {len(soft_solve_rows)}")
    print(f"Rows removed: {len(df) - len(soft_solve_rows)}")


def filter_problem_codes(input_csv, output_csv):
    """Filter CSV to keep only rows with allowed problem codes."""
    df = _load_csv(input_csv)

    # Synthetic schema: fault_category
    pc_col = df["fault_category"] if "fault_category" in df.columns else df["u_problem_code"]
    filtered_df = df[pc_col.isin(ALLOWED_PROBLEM_CODES)]

    filtered_df.to_csv(output_csv, index=False)
    print(f"Original rows: {len(df)}")
    print(f"New rows: {len(filtered_df)}")
    print(f"Rows removed: {len(df) - len(filtered_df)}")


def filter_close_codes(input_csv, output_csv):
    """Filter CSV to keep only rows with allowed close codes."""
    df = _load_csv(input_csv)

    # Synthetic schema: resolution_method
    res_col = df["resolution_method"] if "resolution_method" in df.columns else df["close_code"]
    filtered_df = df[res_col.isin(ALLOWED_CLOSE_CODES)]

    filtered_df.to_csv(output_csv, index=False)
    print(f"Original rows: {len(df)}")
    print(f"New rows: {len(filtered_df)}")
    print(f"Rows removed: {len(df) - len(filtered_df)}")


def main():
    parser = argparse.ArgumentParser(description="Filter incident CSV data based on specific rules.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/anonymized-Incidents_Last_6_Months.csv",
        help="Path to the input CSV file containing incident data. Default: data/anonymized-Incidents_Last_6_Months.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/filtered_file.csv",
        help="Path to save the filtered CSV file. Default: data/filtered_file.csv",
    )
    parser.add_argument("--filter_type", type=str, default="auto")
    args = parser.parse_args()

    # Run the filtering process
    if args.filter_type == "auto":
        filter_auto(args.input_csv, args.output_csv)
    elif args.filter_type == "soft_solve":
        filter_soft_solve(args.input_csv, args.output_csv)
    elif args.filter_type == "problem_codes":
        filter_problem_codes(args.input_csv, args.output_csv)
    elif args.filter_type == "close_codes":
        filter_close_codes(args.input_csv, args.output_csv)
    else:
        parser.error(
            f"Unknown filter_type: {args.filter_type!r}. Choose from: auto, soft_solve, problem_codes, close_codes"
        )


if __name__ == "__main__":
    main()
