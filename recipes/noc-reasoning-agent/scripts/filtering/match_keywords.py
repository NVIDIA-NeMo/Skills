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
import re

import pandas as pd
from tqdm import tqdm


def get_close_codes():
    """
    Parses the structured close code data into a dictionary of categories and keywords.
    """
    # This list contains your full set of close codes and their types (HW, SW, NA)
    raw_data = """
    close_code,Type
    Other,NA
    Alarm Cleared,N/A
    Auto Recovered,NA
    Auto Recover,NA
    Configuration Fixed,SW
    Power Restored,N/A
    IP Configuration Corrected,SW
    Commercial Power Restored,SW
    Cold Reboot,SW
    Reset RU,SW
    Configuration corrected,SW
    Vendor Hardware Replaced,HW
    Auto-Recovered,SW
    Auto Reloaded,SW
    Auto Restart,SW
    Reset DU,SW
    Software Fix,SW
    Network Fix,N/A
    MOP Corrected,SW
    RU Reset,SW
    WCS Fix,N/A
    Restart VM,SW
    Tool Fix,SW
    Fiber Repaired/Replaced,HW
    Change Configuration / parameters,SW
    Corrected Software Path,SW
    Reset ORU,SW
    Antenna Cable  Check,HW
    Rebooted Chassis,SW
    Performance Improvement,SW
    Power Supply Restored,N/A
    Corrected Configuration Error,SW
    Cell Enable / Disable,SW
    Vendor Hardware Repaired,HW
    Tilt Changes,SW
    Activity Completed,N/A
    No Trouble Found,NA
    Closed/Resolved by Caller,SW
    Corrected Config Mismatch,SW
    Restarted Chassis,SW
    Not Within Coverage,NA
    Restarted Process,SW
    SFP Cable cleaned,HW
    GNSS Fix,SW
    KPI Verified,SW
    DU Reboot,SW
    Replaced Antenna,HW
    Site Restored,N/A
    Issue Fixed,N/A
    Corrected wiring,HW
    Replaced controller board,HW
    Firmware upgrade,SW
    GPS Cone Replacement,HW
    Non-RF Issue,NA
    Poor Indoor Coverage,N/A
    Full Time Roaming,N/A
    Rebooted NNI,SW
    Hardware Replaced,HW
    Cleaned Fiber,HW
    Restarted ueconmgr,SW
    Water Remediation,HW
    Restart MVRP Services,SW
    Updated Configuration,SW
    SFP Replaced,SW
    Payment completed,NA
    Delete DU Pod,SW
    New Site Deployment,HW
    Device Changed,HW
    Restarted gnbmgr,SW
    Replaced SFP,HW
    Sotware Bug Fix,SW
    Cell locked / Unlocked,SW
    Delete M-Plane and F1C IP,SW
    Rebooted ESXi Host,SW
    Replaced RU,HW
    PowerCycle CSR,N/A
    CU reset,HW
    No Impact,NA
    Cleared Disk space,SW
    Replaced Hardware,HW
    Unnotified Carrier Maintenance,NA
    Restarted Network Card,SW
    Replaced BMC Server,HW
    Delete PTP Pod,SW
    Replaced complete unit,HW
    Delete MTCIL Pod,SW
    Firmware/Software Configuration,SW
    DU Replacement,HW
    RET Adjustment,SW
    Replaced Power Supply,HW
    improved by Optimization,SW
    Reinstantiated Site/CICD,N/A
    Cancelled,NA
    Replaced/repaired external cable,HW
    Rolled Back Change,SW
    Fiber/Connector Repaired/Replaced,HW
    POD Reboot,SW
    GPS Replaced,HW
    GPS Cable replaced,HW
    Application Reinstantiated,SW
    Copper/Optic Cable Replaced,HW
    Access Realignment,SW
    Pods Restarted,SW
    Memory adjusted,SW
    DU Reset,SW
    Delete Core Files,SW
    Fiber Cable replaced,HW
    BMC Reset,SW
    Repaired connector,HW
    Fiber Replaced,HW
    Device Configuration corrected,SW
    Power Equipment Repaired,HW
    RU Replacement,HW
    Hardware Replace,HW
    Replaced Optical Card,HW
    Re provisioned,SW
    Replaced/Reseated Cabled,SW
    Hybrid replaced,N/A
    DU,N/A
    Replaced Fan Module,HW
    Fiber Replacement,HW
    Replaced CSR,HW
    Rebooted NID,SW
    Sleepy CU-PODs restarted,SW
    Electronic components replaced,HW
    SFP/cable Replaced,HW
    Repaired Cable/Connector,HW
    Cleared by Government Officials,NA
    Configuration Corrected/Updated,SW
    Replaced GPS antenna,HW
    Deleted Prior EC2,SW
    Host Restart,SW
    Software Stability,SW
    Rerun CICD Pipeline,SW
    Repalced the connector cables,HW
    CSR Power Supply Replaced,HW
    Restarted Manually,HW
    Corrected plungers,HW
    Restart Application,SW
    RU/DU Restart,SW
    Pod Reset,SW
    CUCP Pod Swtichover to worker node,SW
    HVAC repaired/replaced,HW
    Cleaned Fan Filter,HW
    SFP Cable Replaced,HW
    Others,N/A
    Solved Remotely (Permanently),SW
    Replaced Network Card,HW
    FE Rebooted,SW
    Network Switch,N/A
    Reset RET's,SW
    Replaced Attenuator,HW
    Power Cycle,N/A
    Rollout Restart,N/A
    Replaced Chassis,HW
    Re-Deployment,SW
    Initialized PODS,SW
    Replaced NID,HW
    CPU Replaced,HW
    Corrected BGP Configuration,SW
    Replaced the connector cables,HW
    bccsvc Restart,SW
    Rebooted Network Card,SW
    Delete MVRP Logs,SW
    DC Power Bounce,SW
    POD Reboot - USM,SW
    BMC Power Supply Replaced,HW
    Reboot Dpp Pod,SW
    Capacity Increase,SW
    Sleepy CUCP restarted,SW
    Hardware Restarted,SW
    Cleared/Reformatted Disk 0,SW
    Replaced /repaired internal cable,HW
    Groundbars/Copper replaced,HW
    Improved by Parameter Change,SW
    Adding sdaas_ip incorrect in BPI or infoblox,SW
    Fiber Sweep,HW
    Restarted ngclientiwf,SW
    Replaced fans,HW
    Reconfigured BMC,SW
    eCPRI Fiber Replaced,HW
    Generator Deployement,HW
    Restored Tripped Breaker,HW
    Batteries Replaced,HW
    Not a Sleepy CU,SW
    Cleared/Reformatted Disk 1,SW
    Reserved / Unreserved,N/A
    Door swap,HW
    RET motor replacement,HW
    Fiber Connectivity Restored,N/A
    Initialize Mplane,SW
    Replaced polyphaser,HW
    RU Software Reset,SW
    Barred/ Not Barred,SW
    Cabling Replaced,HW
    Replaced RET cable,HW
    Restarted sctpe1iwf,SW
    Activated OCNS,SW
    VM Reset,SW
    Restarted bccsvc,SW
    SMF restart,SW
    RU software bug fix,SW
    Cleared In Testing,NA
    Solar Power Restored,SW
    Restarted sctpf1iwf,SW
    Updated Lat/long settings,SW
    Restarted gwsvc,SW
    Cabinet Replacement,HW
    Disk Clean,SW
    NID Replaced,HW
    """

    from io import StringIO

    codes_df = pd.read_csv(StringIO(raw_data))

    categorized_codes = {
        "Hardware": [code.lower().replace(" ", "") for code, type in codes_df.values if type == "HW"],
        "Software": [code.lower().replace(" ", "") for code, type in codes_df.values if type == "SW"],
    }
    return categorized_codes


# Synthetic data uses different resolution_method values. Map them to physical_intervention vs soft_solve.
SYNTHETIC_RESOLUTION_TO_CATEGORY = {
    "physical_intervention": [
        "field dispatch required",
        "escalated",
    ],
    "soft_solve": [
        "resolved",
        "issue corrected",
        "service restored",
        "partial resolution",
        "pending resolution",
    ],
}


def find_keyword_matches(row, pattern, keywords):
    """Finds which specific keywords from a list match within a DataFrame row."""
    # Synthetic schema uses resolution_method (same meaning as close_code)
    resolution_method = row.get("resolution_method", row.get("close_code", ""))
    if resolution_method and str(resolution_method).lower().replace(" ", "") in keywords:
        return True

    return False


def categorize_incidents_by_close_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes incidents into 'Hardware' or 'Software' based on a structured list of close codes.
    It prioritizes Hardware matches over Software matches.
    """
    # Ensure required output columns exist
    if "solved_category" not in df.columns:
        df["solved_category"] = "Uncategorized"
    if "solved_reason" not in df.columns:
        df["solved_reason"] = ""

    # Get the categorized lists of keywords
    close_codes = get_close_codes()
    hw_keywords = close_codes["Hardware"]
    sw_keywords = close_codes["Software"]

    # Precompile regex patterns for performance. This matches any of the phrases.
    hw_pattern = re.compile("|".join(re.escape(k) for k in hw_keywords), flags=re.IGNORECASE)
    sw_pattern = re.compile("|".join(re.escape(k) for k in sw_keywords), flags=re.IGNORECASE)

    # Convert all data to string type for safe searching
    str_df = df.astype(str)

    print("Categorizing incidents based on close codes...")
    for idx, row in tqdm(str_df.iterrows(), total=len(df)):
        # Prioritize Hardware: Check for HW keywords first
        hw_matches = find_keyword_matches(row, hw_pattern, hw_keywords)
        if hw_matches:
            df.at[idx, "solved_category"] = "physical_intervention"
            continue  # Move to the next row once categorized

        # If no HW keywords, check for SW keywords
        sw_matches = find_keyword_matches(row, sw_pattern, sw_keywords)
        if sw_matches:
            df.at[idx, "solved_category"] = "soft_solve"
            continue

        # Synthetic data: resolution_method values not in legacy close-code list
        resolution_method = row.get("resolution_method", row.get("close_code", ""))
        if resolution_method:
            rm_normalized = str(resolution_method).strip().lower()
            for category, values in SYNTHETIC_RESOLUTION_TO_CATEGORY.items():
                if rm_normalized in values:
                    df.at[idx, "solved_category"] = category
                    break

    hw_count = (df["solved_category"] == "physical_intervention").sum()
    sw_count = (df["solved_category"] == "soft_solve").sum()
    un_count = (df["solved_category"] == "Uncategorized").sum()

    print("\n--- Categorization Complete ---")
    print(f"Total rows processed: {len(df)}")
    print(f"Physical Intervention: {hw_count}")
    print(f"Soft Solve: {sw_count}")
    print(f"Uncategorized: {un_count}")

    return df


if __name__ == "__main__":
    # Example: Adjust these paths to your actual data files

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
        default="data/categorized_incidents.csv",
        help="Path to save the filtered CSV file. Default: data/categorized_incidents.csv",
    )
    args = parser.parse_args()

    input_file = args.input_csv
    output_file = args.output_csv

    print(f"Loading data from {input_file}...")
    # Use 'latin1' encoding if your CSV has special characters
    df = pd.read_csv(input_file, encoding="latin1")

    # Run the categorization function
    df = categorize_incidents_by_close_code(df)

    # Save the updated dataframe to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"\nCategorized data saved to: {output_file}")
