import pandas as pd


def _safe_tool_val(val, default: str = "NotApplicable") -> str:
    """Return a string safe for JSON; pandas NaN and None become default."""
    if val is None:
        return default
    if isinstance(val, float) and (val != val or val == float("inf") or val == float("-inf")):
        return default
    return str(val)


def Check_Alarm_Status(row: pd.Series, site_or_element_id: str) -> str:
    """
    Retrieves current alarm details, severity, and active time for a given site or network element.

    Input: site_or_element_id (e.g., 'Site-A', 'DU-123')
    Output: Alarm status (active/cleared), severity, timestamp.
    """
    return row.get("Check_Alarm_Status", "NotApplicable")


def Check_Element_Neighbors(row: pd.Series, element_id: str) -> str:
    """
    Checks all adjacent and upstream devices of a target element to find common alarms affecting the area.

    Input: element_id
    Output: List of adjacent elements with active alarms or status.
    """
    return _safe_tool_val(row.get("Check_Element_Neighbors", "NotApplicable"))


def Check_Element_Health(row: pd.Series, element_id: str) -> str:
    """
    Polls the element (e.g., DU/RU) to retrieve key health metrics like cell status and radiation.

    Input: element_id
    Output: Health metrics report (e.g., Cell Status: UP, Radiation: Normal).
    """
    return _safe_tool_val(row.get("Check_Element_Health", "NotApplicable"))


def Execute_Remote_Action(row: pd.Series, element_id: str, action: str) -> str:
    """
    Runs a specific remote command on an element.
    Example Actions: 'unlock_cell', 'restart_du', 'restore_ru'

    Input: element_id, action
    Output: Execution result (Success/Fail).
    """
    return _safe_tool_val(row.get("Execute_Remote_Action", "NotApplicable"))


def Check_External_Issues(row: pd.Series, site_or_area: str) -> str:
    """
    Scans external monitors (like DownDetector or topology maps) for area-wide issues like fiber cuts or power outages.

    Input: site_or_area
    Output: External issue report (e.g., Fiber cut detected in region).
    """
    return _safe_tool_val(row.get("Check_External_Issues", "NotApplicable"))


def Check_Apply_Configuration(row: pd.Series, element_id: str) -> str:
    """
    Retrieves the element's configuration, validates it against the standard, and pushes a corrected config if a mismatch is found.

    Input: element_id
    Output: Config validation result and application status.
    """
    return _safe_tool_val(row.get("Check_Apply_Configuration", "NotApplicable"))


def Check_Performance(row: pd.Series, kpi_metric_name: str) -> str:
    """
    Fetches a specific KPI from monitoring tools to check if its trends are in line with expectations.
    Example Metric: 'PRACH success rate'

    Input: kpi_metric_name
    Output: KPI trend analysis (e.g., 'PRACH success rate is below threshold').
    """
    return _safe_tool_val(row.get("Check_Performance", "NotApplicable"))


def Create_Ticket(row: pd.Series, department_name: str, issue_details: str) -> str:
    """
    Logs a new issue in the ticketing system and routes it to the correct department.

    Input: department_name, issue_details
    Output: Ticket ID and routing confirmation.
    """
    return _safe_tool_val(row.get("Create_Ticket", "NotApplicable"))


def Orchestration_tool(row: pd.Series, action_command: str) -> str:
    """
    Runs an automated O-RAN orchestration task using Kubernetes/Helm.
    Example Action: 'delete_pod_xyz', 'reassign_ip_address'


    Input: action_command
    Output: Orchestration task status.
    """
    return _safe_tool_val(row.get("Orchestration_tool", "NotApplicable"))


def Triage_Toolkit_Tool(row: pd.Series, issue_type: str) -> str:
    """
    Executes diagnostic scripts specifically for container or pod-related issues.
    Example Issue Type: 'pod-crash-loop', 'container-networking'

    Input: issue_type
    Output: Diagnostic logs and root cause hints.
    """
    return _safe_tool_val(row.get("Triage_Toolkit_Tool", "NotApplicable"))


def Check_remote_files(row: pd.Series, element_id: str) -> str:
    """
    Connects to a device via SSH/Telnet to review system dump files for identified errors or issues.

    Input: element_id
    Output: Analysis of dump files (e.g., 'Memory overflow error found').
    """
    return _safe_tool_val(row.get("Check_remote_files", "NotApplicable"))


ALL_TOOLS = [
    Check_Alarm_Status,
    Check_Element_Neighbors,
    Check_Element_Health,
    Execute_Remote_Action,
    Check_External_Issues,
    Check_Apply_Configuration,
    Check_Performance,
    Create_Ticket,
    Orchestration_tool,
    Triage_Toolkit_Tool,
    Check_remote_files,
]

ALL_TOOLS_STRING = [tool.__name__ for tool in ALL_TOOLS]
