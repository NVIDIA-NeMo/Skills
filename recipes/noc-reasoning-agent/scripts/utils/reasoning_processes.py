PROBLEM_CODE_REASONING_PROCESS = {
    "Service-off": """
    1. Acknowledge and Track the Initial Alert: Reference site/alarm number. Explanation: This step logs the alert receipt and assigns a tracking ID to initiate the response process.
        - Tools Used: Check_Alarm_Status() - Queries the alarm system for details.
        - Flow-Deciding Answers/Outcomes: If alarm confirmed active, proceed to step 2; if cleared/false positive, document and close.
    2. Check Site and Equipment Status: Note locked/disabled components. Explanation: This step assesses current site conditions to identify immediate visible issues like equipment locks.
        - Tools Used: Check_Element_Health() - Polls health metrics.
        - Flow-Deciding Answers/Outcomes: If anomalies are detected (e.g., locked cells), go to step 3; if healthy, monitor and close.
    3. Perform Remote Actions (Reset/Unlock/Enable): Use CLI/management tools. Explanation: This step attempts non-invasive fixes like resetting components to restore functionality remotely.
        - Tools Used: Remote_Unlock/Restart/Restore(Element) - Executes resets.
        - Flow-Deciding Answers/Outcomes: If action succeeds, go to step 4; if fails, escalate to step 8.
    4. Monitor for Automatic Recovery or Alarm Clearance: Confirm health. Explanation: This step observes post-action behavior to see if the system self-recovers without further intervention.
        - Tools Used: Check_Element_Health() - Re-check post-action.
        - Flow-Deciding Answers/Outcomes: If recovered, go to step 8; if not, proceed to step 5.
    5. Check Topology/Outages/Fiber Cut/Public Websites: Identify external issues. Explanation: This step investigates network-wide or external factors impacting the site.
        - Tools Used: Check_External_Issues() - Scans for outages.
        - Flow-Deciding Answers/Outcomes: If external issues are found, assigned to relevant departments and close; else, go to step 6.
    6. Check/Apply Configuration Fixes: Correct mismatches (e.g., IP/VLAN). Explanation: This step audits and corrects software settings to align with operational standards.
        - Tools Used: Check_Apply_Configuration(Element) - Validates and pushes fixes.
        - Flow-Deciding Answers/Outcomes: If fixed, go to step 7; if not, escalate to step 8.
    7. Investigate Persistent or Reoccurring Alarms and Coordinate Onsite Dispatch if Needed/ or Hardware Replacement: Dispatch teams. Explanation: This step mobilizes deeper research and physical intervention when remote fixes are insufficient.
        - Tools Used: Create_Ticket(Department_Name) - Routes ticket.
        - Flow-Deciding Answers/Outcomes: After onsite resolution, go to step 8.
    8. Confirm Final Site Health and Close the Incident: Ensure stability. Explanation: This step verifies full restoration before officially ending the incident.
        - Tools Used: Check_Alarm_Status(); Check_Element_Health() - Final verification.
        - Flow-Deciding Answers/Outcomes: If healthy, document and close; else, loop to step 7.
    9. Document All Actions and Status Updates: Record steps. Explanation: This step compiles a complete record for audits, learning, and future reference.
        - Tools Used: None.
        - Flow-Deciding Answers/Outcomes: Always close after this.
    """,
    "Degraded Prach": """
    1. Identify and Categorize the Alarm Event. Explanation: This step classifies the alarm to determine the appropriate troubleshooting path.
        - Tools Used: Check_Alarm_Status() - Categorizes alarm type.
        - Flow-Deciding Answers/Outcomes: If PRACH-specific, proceed; if unrelated, reassign and close.
    2. Gather Site and Equipment Status: Include DU/RU/cell health. Explanation: This step collects baseline data to pinpoint degradation sources.
        - Tools Used: Check_Elements_Health(Neighbors) - Collects status from Neighbors or upstream devices.
        - Flow-Deciding Answers/Outcomes: If issues identified, go to step 3; if normal, monitor and close.
    3. Perform Remote Checks and Initial Actions: Reset if safe. Explanation: This step conducts preliminary diagnostics and minor fixes remotely.
        - Tools Used: Check_Element_Health() - Initial diagnostics.
        - Flow-Deciding Answers/Outcomes: If basic reset is viable, proceed; else, go to step 8.
    4. Apply Targeted Fixes (e.g., Unlock Cells, Correct Configs). Explanation: This step implements specific corrections based on identified issues.
        - Tools Used: Check_Apply_Configuration(); - Checks and Corrects Config..
        - Flow-Deciding Answers/Outcomes: If fix Config went well, go to step 5; else, go to step 8.
    5. Restart or Reinitialize Network Elements: Reboot DU/RU/pods. Explanation: This step cycles components to clear transient errors.
        - Tools Used: Remote_Unlock/Restart/Restore() - Reinitializes.
        - Flow-Deciding Answers/Outcomes: If successful, go to step 6; else, escalate.
    6. Monitor for Recovery and Stability: Check PRACH performance. Explanation: This step tracks improvements post-fix to ensure sustained resolution.
        - Tools Used: Check_Element_Health() - Monitors post-fix.
        - Flow-Deciding Answers/Outcomes: If stable, go to step 7; else, go to step 8.
    7. Validate with KPI and Alarm Dashboards: Confirm metrics. Explanation: This step uses data to objectively verify fix effectiveness.
        - Tools Used: Check_Performancel() - Fetches KPIs.
        - Flow-Deciding Answers/Outcomes: If KPIs normal, document and close; else, loop to step 4.
    8. Assign the Incident if not solved: To technical/field team. Explanation: This step transfers unresolved issues to specialized teams.
        - Tools Used: Create_Ticket() - Routes ticket.
        - Flow-Deciding Answers/Outcomes: After resolution, go to step 9.
    9. Document and Close the Incident: Record steps/outcomes. Explanation: This step finalizes the record for compliance and knowledge sharing.
        - No tool used.
        - Flow-Deciding Answers/Outcomes: Always close.

    """,
    "Offline / Unreachable": """
    1. Check Alarm Status. Explanation: This step confirms the alarm's validity to avoid false positives.
        - Tools Used: Check_Alarm_Status() - Verifies alarm.
        - Flow-Deciding Answers/Outcomes: If active, proceed; else, close.
    2. RU Reset: Restart to clear issues. Explanation: This step attempts a basic restart to resolve connectivity.
        - Tools Used: Check_Alarm_Status(in RU) - Verifies alarm.
        - Flow-Deciding Answers/Outcomes: If active, proceed; else, go to step 3.
        - Tools Used: Remote_Unlock/Restart/Restore(RU) - Resets RU.
        - Flow-Deciding Answers/Outcomes: If cleared, go to step 9; else, proceed.
    3. TX Array Alarm Clearance: Restart TX array/RU. Explanation: This step targets transmission-specific alarms for clearance.
        - Tools Used: Check_Alarm_Status(in TX) - Verifies alarm.
        - Flow-Deciding Answers/Outcomes: If active, proceed; else, go to step 4.
        - Tools Used: Remote_Unlock/Restart/Restore(TX Array) - Clears alarm.
        - Flow-Deciding Answers/Outcomes: If resolved, go to step 9; else, proceed.
    4. CSR Power Restoration: Reset Router. Explanation: This step addresses power-related unreachable states.
        - Tools Used: Remote_Unlock/Restart/Restore(Router) - Triggers reset.
        - Flow-Deciding Answers/Outcomes: If restored, go to step 9; else, proceed.
    5. Digital Input Low Alarm WA Application: Apply workaround. Explanation: This step implements temporary fixes for input signal issues.
        - Tools Used: Remote_Unlock/Restart/Restore(TSSI) - Applies WA.
        - Flow-Deciding Answers/Outcomes: If fixed, go to step 9; else, proceed.
    6. OranLoginFailure Recovery: Fix login issues in RUs. Explanation: This step resolves authentication failures preventing access.
        - Tools Used: Remote_Unlock/Restart/Restore(RU) - Recovers login.
        - Flow-Deciding Answers/Outcomes: If resolved, go to step 9; else, proceed.
    7. BGP Flap Resolution: Stabilize neighbors. Explanation: This step fixes routing instability, causing unreachability.
        - Tools Used: Check_Element_Health(BGP Neighbors) - Resolves flaps.
        - Flow-Deciding Answers/Outcomes: If stable, go to step 9; else, proceed.
    8. ISIS Adjacency Recovery: Restore links. Explanation: This step reestablishes routing adjacencies.
        - Tools Used: Check_Element_Health(Adjacencies) - Recovers adjacency.
        - Flow-Deciding Answers/Outcomes: If restored, go to step 9; else, go to step 10.
    9. Cell Enablement: Enable after resets. Explanation: This step activates cells post-recovery to restore service.
        - Tools Used: Check_Apply_Configuration(Cell);
        - Check_Element_Health() - Enables cells.
        - Flow-Deciding Answers/Outcomes: If enabled, document and close; else, escalate.
    10. Escalate for Onsite: Dispatch teams. Explanation: This step initiates physical checks when remote options fail.
        - Tools Used: Create_Ticket() - Routes ticket.
        - Flow-Deciding Answers/Outcomes: After resolution, document and close.
    11. Document and Close: Record steps. Explanation: This step ensures a complete audit trail for the incident.
        - Tools Used: None.
        - Flow-Deciding Answers/Outcomes: Always close.

    """,
    "Disabled Cells": """
    1. Check Alarm Status. Explanation: This step validates the alarm to confirm cell disablement.
        - Tools Used: Check_Alarm_Status() - Verifies alarm.
        - Flow-Deciding Answers/Outcomes: If active, proceed; else go to step 8.
    2. Enable Cell: Check if the cell can be enabled by applying a config change.. Explanation: This step manually activates disabled cells.
        - Tools Used: Check_Apply_Configuration(); Enables cells.Flow-Deciding Answers/Outcomes: If the cell can be enabled, go to step 8; else, proceed.
    3. RU Reset: Restart RU. Explanation: This step clears potential transient locks via restart.
        - Tools Used: Remote_Unlock/Restart(RU) - Resets RU.
        - Flow-Deciding Answers/Outcomes: If Reset ok, proceed; otherwise, go to step 7.
    4. Check Alarm Status. Explanation: This step validates the alarm to confirm cell disablement.
        - Tools Used: Check_Alarm_Status() - Verifies alarm.
        - Flow-Deciding Answers/Outcomes: If active, proceed; else go to step 8.
    5. Performed Gamma MB RU Reset: Specific reset for issues. Explanation: This step applies a targeted reset for persistent problems.
        - Tools Used: Remote_Unlock/Restart/Restore(RU) - Performs reset.
        - Flow-Deciding Answers/Outcomes: If the reset was ok, proceed; otherwise, go to step 7
    6. Check Alarm Status. Explanation: This step validates the alarm to confirm cell disablement.
        - Tools Used: Check_Alarm_Status() - Verifies alarm.
        - Flow-Deciding Answers/Outcomes: If active, proceed; else go to step 8.
    7. Escalate for Onsite: Dispatch teams. Explanation: This step sends field support for hardware verification.
        - Tools Used: Create_Ticket() - Routes ticket.
        - Flow-Deciding Answers/Outcomes: After resolution, go to step 8.
    8. Document and Close: Record steps. Explanation: This step logs the resolution for records and analysis.
        - Tools Used: None.
        - Flow-Deciding Answers/Outcomes: Always close.
    """,
    "Node Down": """
    1. Check Alarm Status. Explanation: This step confirms the node's down status via alarms.
        - Tools Used: Check_Alarm_Status() - Verifies alarm.
        - Flow-Deciding Answers/Outcomes: If active, proceed; else, close.
    2. Rebooting the Host: Restart machine. Explanation: This step restarts the host to resolve software hangs.
        - Tools Used: Remote_Unlock/Restart/Restore() - Reboots host.
        - Flow-Deciding Answers/Outcomes: If cleared, go to step 10; else, proceed.
    3. Reconfiguring the Site: Update settings. Explanation: This step adjusts site configs to fix misalignments.
        - Tools Used: Check_Apply_Configuration()- Reconfigures.
        - Flow-Deciding Answers/Outcomes: If fixed, go to step 10; else, proceed.
    4. Deleting M-Plane & F1C IP: Remove for recreation. Explanation: This step clears IPs to enable fresh container setup.
        - Tools Used: Orchestration_tool(Delete/Recreate Container) - Deletes/reassigns.
        - Flow-Deciding Answers/Outcomes: If successful, go to step 10; else, proceed.
    5. Using the Triage Toolkit: Troubleshoot containers. Explanation: This step runs diagnostics on container-related issues.
        - Tools Used: Triage_Toolkit_Tool() - Runs diagnostics.
        - Flow-Deciding Answers/Outcomes: If resolved, go to step 10; else, proceed.
    6. Checking SCTP Connection for F1: Verify via netstat. Explanation: This step inspects F1 links for connectivity problems.
        - Tools Used: Check_Element_Health(SCTP ) - Executes netstat returns Ok/Nok
        - Flow-Deciding Answers/Outcomes: If connected, go to step 10; else, proceed.
    7. Verifying RU Reachability: Check DU-RU link. Explanation: This step tests if DU can access RUs.
        - Tools Used: Check_Element_Health(RU) - Verifies reachability.
        - Flow-Deciding Answers/Outcomes: If reachable, go to step 10; else, proceed.
    8. Checking Core Dump Files: Analyze crashes. Explanation: This step reviews dumps for crash insights.
        - Tools Used: Check_remote_files() - Evaluates dumps.
        - Flow-Deciding Answers/Outcomes: If issues are identified/fixed, go to step 10; else, proceed.
    9. Checking Node Uptime/Load Average: Review performance. Explanation: This step evaluates resource usage for overload clues.
        - Tools Used: Check_Performance() - Monitors metrics.
        - Flow-Deciding Answers/Outcomes: If normal, go to step 10; else, go to step 11.
    10. Redeploying the Site: If needed. Explanation: This step rebuilds the site for comprehensive recovery.
        - Tools Used: Check_Apply_Configuration(); - Redeploys.
        - Flow-Deciding Answers/Outcomes: If resolved, document and close; else, escalate.
    11. Escalate for Onsite: Dispatch teams. Explanation: This step engages physical support for unresolved issues.
        - Tools Used:Create_Ticket() - Routes ticket.
        - Flow-Deciding Answers/Outcomes: After resolution, document and close.
    12. Document and Close: Record steps. Explanation: This step creates an audit-ready summary of the incident.
        - Tools Used: None.
        - Flow-Deciding Answers/Outcomes: Always close.
    """,
    "Site Not Scrolling": """

    """,
    "Sleepy Cell": """

    """,
    "VM is in not ready state": """

    """,
    "Prach 0": """

    """,
    "N2 Link Down": """

    """,
    "ueconmgr pod restarted": """

    """,
    "CSR Not Reachable": """

    """,
    "Circuit Down": """

    """,
    "Link Down": """

    """,
    "GPS Sync": """

    """,
    "MTA Alert": """

    """,
}

# Default reasoning process when fault_category (synthetic workflow ID) is not in PROBLEM_CODE_REASONING_PROCESS.
DEFAULT_REASONING_PROCESS = PROBLEM_CODE_REASONING_PROCESS.get("Service-off", "")

# Synthetic data uses fault_category = workflow IDs (e.g. power_ac_failure_recovery). Map them to a reasoning process.
SYNTHETIC_FAULT_CATEGORY_REASONING = {
    "power_ac_failure_recovery": PROBLEM_CODE_REASONING_PROCESS.get("Service-off", ""),
    "power_dc_rectifier_recovery": PROBLEM_CODE_REASONING_PROCESS.get("Service-off", ""),
    "power_battery_discharge_response": PROBLEM_CODE_REASONING_PROCESS.get("Service-off", ""),
    "power_generator_failure_recovery": PROBLEM_CODE_REASONING_PROCESS.get("Service-off", ""),
    "env_high_temperature_response": PROBLEM_CODE_REASONING_PROCESS.get("Service-off", ""),
    "env_hvac_fault_recovery": PROBLEM_CODE_REASONING_PROCESS.get("Service-off", ""),
    "env_water_intrusion_response": PROBLEM_CODE_REASONING_PROCESS.get("Service-off", ""),
}


def get_reasoning_process_for_fault_category(fault_category: str) -> str:
    """Return reasoning process text for synthetic fault_category (or original u_problem_code)."""
    if fault_category is None or str(fault_category).strip() in ("", "nan", "None"):
        return DEFAULT_REASONING_PROCESS
    fc = str(fault_category).strip()
    if fc in SYNTHETIC_FAULT_CATEGORY_REASONING:
        return SYNTHETIC_FAULT_CATEGORY_REASONING[fc]
    if fc in PROBLEM_CODE_REASONING_PROCESS:
        return PROBLEM_CODE_REASONING_PROCESS[fc]
    return DEFAULT_REASONING_PROCESS
