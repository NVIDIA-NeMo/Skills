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

"""
Canonical column names for the pipeline when using synthetic (new) data format.
Use these names everywhere so the pipeline runs on CSV/JSONL with synthetic schema
without converting to the old format.
"""

from typing import List

# Primary key for incidents (CSV column and JSONL key)
INCIDENT_ID_COLUMN = "incident_identifier"

# All columns expected in synthetic CSV (and produced in JSONL).
# Derived columns: time_to_resolve (computed), solved_category (from match_keywords), problem_code_reasoning_process (mapped from fault_category).
REQUIRED_COLUMNS: List[str] = [
    INCIDENT_ID_COLUMN,
    "geographical_territory",
    "incident_classification",
    "incident_subtype",
    "service_domain",
    "equipment_provider",
    "operational_zone",
    "resolution_status",
    "suspension_cause",
    "urgency_level",
    "affected_site",
    "incident_summary",
    "detection_timestamp",
    "escalation_date",
    "responsible_team",
    "fault_category",
    "root_cause_primary",
    "resolution_summary",
    "action_chronicle",
    "reporter_identity",
    "intervention_began",
    "intervention_completed",
    "resolution_method",
    "root_cause_secondary",
    "cause_additional",
    "triggered_by_modification",
    "resolver_identity",
    "time_to_resolve",
    "solved_category",
    "problem_code_reasoning_process",
]

# Column used for close-code / resolution classification (match_keywords, filter_rows)
RESOLUTION_METHOD_COLUMN = "resolution_method"
RESOLUTION_SUMMARY_COLUMN = "resolution_summary"
FAULT_CATEGORY_COLUMN = "fault_category"
SOLVED_CATEGORY_COLUMN = "solved_category"

# Datetime columns (for parsing and time_to_resolve)
INTERVENTION_BEGAN_COLUMN = "intervention_began"
INTERVENTION_COMPLETED_COLUMN = "intervention_completed"

# Prompt/display columns (used in YAML and scripts)
PROMPT_COLUMNS = {
    "incident_identifier": INCIDENT_ID_COLUMN,
    "geographical_territory": "geographical_territory",
    "incident_classification": "incident_classification",
    "incident_subtype": "incident_subtype",
    "service_domain": "service_domain",
    "equipment_provider": "equipment_provider",
    "operational_zone": "operational_zone",
    "urgency_level": "urgency_level",
    "affected_site": "affected_site",
    "incident_summary": "incident_summary",
    "detection_timestamp": "detection_timestamp",
    "responsible_team": "responsible_team",
    "fault_category": "fault_category",
}
