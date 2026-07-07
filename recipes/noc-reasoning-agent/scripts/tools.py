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

import pandas as pd


def _safe_tool_val(val, default: str = "NotApplicable") -> str:
    """Return a string safe for JSON; pandas NaN and None become default."""
    if val is None:
        return default
    if isinstance(val, float) and (val != val or val == float("inf") or val == float("-inf")):
        return default
    return str(val)


def query_alarm(row: pd.Series, site_or_element_id: str = "", **kwargs) -> str:
    """
    Queries the alarm management system to retrieve current alarm details,
    severity, and active time.

    Input: site_or_element_id
    Returns: Alarm status (active/cleared), severity, timestamp, description.
    """
    return _safe_tool_val(row.get("query_alarm", row.get("Check_Alarm_Status", "NotApplicable")))


def query_resource_health(row: pd.Series, element_id: str = "", **kwargs) -> str:
    """
    Polls monitoring systems (EMS/NMS/Telemetry) to retrieve device health
    metrics such as CPU, memory, interface status, and cell state.

    Input: element_id
    Returns: Health report (e.g., 'All systems operational' / 'Fault detected').
    """
    return _safe_tool_val(row.get("query_resource_health", row.get("Check_Element_Health", "NotApplicable")))


def query_performance(row: pd.Series, metric_type: str = "", **kwargs) -> str:
    """
    Fetches KPIs from monitoring tools. Reports if trends are in line with
    expectations or not. Supports metrics like PRB utilization, throughput,
    handover stats, VoLTE KPIs, signaling load, etc.

    Input: metric_type (e.g., 'prb_utilization', 'throughput', 'volte_kpi')
    Returns: KPI status (OK / NOK / trend analysis).
    """
    return _safe_tool_val(row.get("query_performance", row.get("Check_Performance", "NotApplicable")))


def query_topology(row: pd.Series, element_id: str = "", **kwargs) -> str:
    """
    Verifies neighbors, adjacencies, upstream devices, and identifies common
    alarms in an area. Also maps affected services through topology.

    Input: element_id
    Returns: Adjacent elements with alarm status, affected services.
    """
    return _safe_tool_val(row.get("query_topology", row.get("Check_Element_Neighbors", "NotApplicable")))


def execute_remote_action(row: pd.Series, element_id: str = "", action: str = "", **kwargs) -> str:
    """
    Executes remote CLI commands (SSH/Netconf) for resets, restarts,
    failovers, and other operational actions on network elements.

    Input: element_id, action (e.g., 'enodeb_reset', 'generator_start', 'sctp_reset')
    Returns: Execution result (Success/Fail).
    """
    return _safe_tool_val(row.get("execute_remote_action", row.get("Execute_Remote_Action", "NotApplicable")))


def apply_configuration(row: pd.Series, element_id: str = "", config_type: str = "", **kwargs) -> str:
    """
    Retrieves, validates, and pushes configuration changes. Supports
    parameter adjustments, load balancing, HVAC settings, routing changes, etc.

    Input: element_id, config_type (e.g., 'load_balancing', 'hvac_setpoint')
    Returns: Configuration Verified/Applied or error.
    """
    return _safe_tool_val(row.get("apply_configuration", row.get("Check_Apply_Configuration", "NotApplicable")))


def run_diagnostics(row: pd.Series, diagnostic_type: str = "", **kwargs) -> str:
    """
    Runs diagnostic scripts including config audits, OTDR tests, kubelet
    checks, resource usage analysis, and more.

    Input: diagnostic_type (e.g., 'config_audit', 'otdr', 'kubelet')
    Returns: Diagnostic report summary.
    """
    return _safe_tool_val(row.get("run_diagnostics", row.get("Triage_Toolkit_Tool", "NotApplicable")))


def inspect_logs(row: pd.Series, log_type: str = "", **kwargs) -> str:
    """
    Connects to devices or containers and reviews system logs, dump files,
    and event records to identify root cause.

    Input: log_type (e.g., 'container', 'bbu_system', 'routing', 'access_control')
    Returns: Log analysis (issues identified / no issues).
    """
    return _safe_tool_val(row.get("inspect_logs", row.get("Check_remote_files", "NotApplicable")))


def create_trouble_ticket(
    row: pd.Series, priority: str = "", team: str = "", issue_details: str = "", **kwargs
) -> str:
    """
    Logs and routes tickets to departments via the ticketing system.
    Supports priority levels and team-specific routing.

    Input: priority, team, issue_details
    Returns: Ticket ID. Once resolved, outputs the solution used.
    """
    return _safe_tool_val(row.get("create_trouble_ticket", row.get("Create_Ticket", "NotApplicable")))


def verify_recovery(row: pd.Series, element_id: str = "", **kwargs) -> str:
    """
    Final verification step — confirms service restoration, checks alarm
    clearance, and validates operational status.

    Input: element_id or service_id
    Returns: Recovery status (recovered / degraded / ongoing).
    """
    return _safe_tool_val(row.get("verify_recovery", "NotApplicable"))


def query_external_factors(row: pd.Series, site_or_area: str = "", **kwargs) -> str:
    """
    Scans for external factors like weather conditions, fiber cuts, utility
    outages, maintenance windows, and scheduled work.

    Input: site_or_area
    Returns: External issue report.
    """
    return _safe_tool_val(row.get("query_external_factors", row.get("Check_External_Issues", "NotApplicable")))


def orchestrate_workload(row: pd.Series, action: str, workload_type: str = "", **kwargs) -> str:
    """
    Automates container/VM operations via Kubernetes or orchestrator:
    restart, scale, migrate, failover, drain, etc.

    Input: action (e.g., 'restart', 'scale', 'migrate'), workload_type (e.g., 'pod', 'vm', 'cnf')
    Returns: Operation status (Successful/Unsuccessful).
    """
    return _safe_tool_val(row.get("orchestrate_workload", row.get("Orchestration_tool", "NotApplicable")))


def query_power_system(row: pd.Series, target: str = "", **kwargs) -> str:
    """
    Queries power infrastructure status including UPS, battery, generator,
    rectifier, and HVAC systems.

    Input: target (e.g., 'ups', 'battery', 'generator', 'hvac', 'rectifier')
    Returns: Power system status and readings.
    """
    return _safe_tool_val(row.get("query_power_system", "NotApplicable"))


def query_rf_status(row: pd.Series, sector_or_antenna_id: str = "", **kwargs) -> str:
    """
    Queries RF chain status including antenna health, PA status, VSWR,
    beamforming, RET controller, and signal measurements.

    Input: sector_or_antenna_id
    Returns: RF status report.
    """
    return _safe_tool_val(row.get("query_rf_status", "NotApplicable"))


def query_container_status(row: pd.Series, container_type: str = "", **kwargs) -> str:
    """
    Queries Kubernetes pod/node/container state from the orchestrator.

    Input: container_type (e.g., 'pod', 'node', 'control_plane', 'cnf', 'service_mesh')
    Returns: Container/pod state details.
    """
    return _safe_tool_val(row.get("query_container_status", "NotApplicable"))


def verify_signaling_path(row: pd.Series, interface: str = "", **kwargs) -> str:
    """
    Tests signaling protocol paths: SCTP, SIP, Diameter, GTP, SIGTRAN,
    and other control plane interfaces.

    Input: interface (e.g., 'SCTP', 'SIP', 'GTP', 'Diameter')
    Returns: Path status (healthy / degraded / down).
    """
    return _safe_tool_val(row.get("verify_signaling_path", "NotApplicable"))


def test_connectivity(row: pd.Series, test_type: str = "", **kwargs) -> str:
    """
    Tests IP/network connectivity: ICMP ping, backhaul, CPRI link,
    peer connectivity, end-to-end path quality.

    Input: test_type (e.g., 'backhaul', 'icmp', 'cpri', 'peer_connectivity')
    Returns: Connectivity test results.
    """
    return _safe_tool_val(row.get("test_connectivity", "NotApplicable"))


ALL_TOOLS = [
    query_alarm,
    query_resource_health,
    query_performance,
    query_topology,
    execute_remote_action,
    apply_configuration,
    run_diagnostics,
    inspect_logs,
    create_trouble_ticket,
    verify_recovery,
    query_external_factors,
    orchestrate_workload,
    query_power_system,
    query_rf_status,
    query_container_status,
    verify_signaling_path,
    test_connectivity,
]

ALL_TOOLS_STRING = [tool.__name__ for tool in ALL_TOOLS]
