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

import html
import re
import sys
from pathlib import Path

import pandas as pd


def parse_steps(reasoning):
    """Parses the Thought/Action/Observation steps from the reasoning trace."""
    step_re = re.compile(r"^(Thought|Action|Observation)\s+(\d+):\s*(.*)$", re.MULTILINE)
    finish_re = re.compile(r"^Finish\[(.*)\]$", re.MULTILINE | re.DOTALL)

    steps = {}
    for kind, num, text in step_re.findall(reasoning or ""):
        steps.setdefault(int(num), {})[kind.lower()] = text.strip()

    ordered_steps = [{"index": i, **steps[i]} for i in sorted(steps)]
    finish_match = finish_re.search(reasoning or "")
    finish_text = finish_match.group(1).strip() if finish_match else None

    return ordered_steps, finish_text


def parse_final_reasoning(generation_text):
    """Parse the 'generation' field to extract the final Thought, Action, and Observation steps."""
    trace_block_match = re.search(
        r"Question:.*?(Finish\[.*?\])",
        generation_text,
        re.DOTALL,
    )

    if trace_block_match:
        reasoning_trace = trace_block_match.group(0)
    else:
        last_index = generation_text.rfind("Finish")
        if last_index != -1:
            reasoning_trace = generation_text[last_index:].strip()
        else:
            return "Final reasoning trace not found."

    step_pattern = re.compile(
        r"^(Thought|Action|Observation)\s+\d+:\s*(.*)$",
        re.MULTILINE,
    )
    steps = step_pattern.findall(reasoning_trace)

    parsed_steps = [f"**{kind}:** {content}" for kind, content in steps]
    return "\n".join(parsed_steps) if parsed_steps else reasoning_trace


def find_finish_action(generation_text):
    """Extract the Finish[...] action text from a generation string."""
    last_index = generation_text.rfind("Finish")
    if last_index != -1:
        return generation_text[last_index:].strip()
    return ""


def parse_generation(generation_text):
    """
    Extracts the clean, final reasoning trace from the raw 'generation' field.
    The trace starts with 'Question:' and ends with 'Finish[...]'.
    """
    # Regex to find the block starting with "Question:" and ending with "Finish[...]"
    trace_re = re.compile(r"Question:.*Finish\[.*\]", re.DOTALL)
    match = trace_re.search(generation_text or "")
    return match.group(0).strip() if match else ""


def parse_work_notes(work_notes_text):
    """Splits work notes into a list of entries based on timestamps."""
    if not work_notes_text:
        return []

    # This regex captures the full timestamp line (e.g., "2024-11-22 10:37:54 - ST (Work notes)")
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s-.*?\))"

    # Split the text by the timestamp pattern, keeping the timestamps
    parts = re.split(timestamp_pattern, work_notes_text)

    notes = []
    # The first item is usually empty, so we start from the first captured timestamp
    i = 1
    while i < len(parts):
        timestamp = parts[i].strip()
        note_text = parts[i + 1].strip()
        if note_text:  # Only add entries that have content
            notes.append({"timestamp": timestamp, "note": note_text})
        i += 2

    return notes


def esc(s):
    """Helper function for HTML escaping."""
    return html.escape(str(s) if s is not None else "")


def render(incident_data):
    """Renders a single incident into a collapsible HTML section."""
    reasoning_trace = incident_data.get("generation")
    finish_action = find_finish_action(incident_data.get("generation"))

    # --- Part 1: Human-Readable Incident Trace ---
    trace_html = "<table>"
    fields_to_display = [
        "incident_identifier",
        "urgency_level",
        "incident_classification",
        "incident_subtype",
        "responsible_team",
        "fault_category",
        "detection_timestamp",
        "incident_summary",
        "geographical_territory",
        "service_domain",
        "equipment_provider",
        "operational_zone",
        "affected_site",
        "escalation_date",
        "generation_start_time",
        "generation_end_time",
        "time_to_resolve",
        "solved_category",
        "solved_reason",
    ]
    for field in fields_to_display:
        value = incident_data.get(field)
        display_value = value if value and str(value).strip() else "—"
        field_name = field.replace("u_", "").replace("_", " ").title()
        trace_html += f'<tr><td class="field-name">{field_name}</td><td>{esc(display_value)}</td></tr>'
    trace_html += "</table>"

    # --- Part 2: Chronological Work Notes ---
    work_notes = incident_data.get("action_chronicle") or incident_data.get("work_notes")
    work_notes_entries = parse_work_notes(work_notes)
    notes_html = "<div class='work-notes-container'>"
    for entry in work_notes_entries:
        notes_html += f"""
        <div class="work-note-entry">
            <div class="timestamp">{esc(entry["timestamp"])}</div>
            <pre class="note-text">{esc(entry["note"])}</pre>
        </div>
        """
    notes_html += "</div>"

    steps, finish = parse_steps(reasoning_trace)
    steps_html = []
    for s in steps:
        block = [f'<div class="step-num">Step {s["index"]}</div>']
        for key, cls in [("thought", "thought"), ("action", "action"), ("observation", "obs")]:
            if s.get(key):
                block.append(f'<div class="{cls}"><b>{key.title()}:</b> {esc(s[key])}</div>')
        steps_html.append(f"<div class='step'>{''.join(block)}</div>")

    # --- Assemble the final collapsible report ---
    incident_id = esc(incident_data.get("incident_identifier", incident_data.get("number", "Unknown Incident")))
    short_desc = esc(incident_data.get("incident_summary", incident_data.get("short_description", "")))

    return f"""
    <details class="incident-details">
      <summary>
        <span class="incident-id">{incident_id}</span>
        <span class="short-desc">{short_desc}</span>
        <ul>
            <li>Category: {esc(incident_data.get("incident_classification", incident_data.get("category")))}</li>
            <li>Problem Code: {esc(incident_data.get("fault_category", incident_data.get("u_problem_code")))}</li>
        </ul>
      </summary>
      <div class="incident-content">
        <h3>Incident Trace (Human-Readable)</h3>
        {trace_html}
        <h3>Work Notes (Chronological)</h3>
        {notes_html}
        <h3>Full Reasoning Trace</h3>
            <div class='work-notes-container'>{reasoning_trace}</div>
        <h3>Thoughts, Observations, Actions</h3>
        <div class="steps">
            {"".join(steps_html)}
        </div>
        <h3> Closing Notes</h3>
            <div class='work-notes-container'>{finish_action}</div>
      </div>
    </details>
    """


def main(input_file, output_file, max_incidents=15, selected_criteria=None):
    """Main function to read, process, and write the HTML report."""
    try:
        # 1. Load the entire dataset using pandas
        df = pd.read_json(input_file, lines=True)
        # Prepare for complex filters by adding helper columns
        notes_col = "action_chronicle" if "action_chronicle" in df.columns else "work_notes"
        df["work_notes_len"] = df[notes_col].astype(str).str.len().fillna(0)
        print(f"Loaded {len(df)} incidents from {input_file}")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return
    except ValueError:
        print(f"Error: Could not parse {input_file}. Ensure it's a valid .jsonl file.")
        return

    if selected_criteria:
        filtered_df = df.query(f"category == '{selected_criteria}'")
    else:
        filtered_df = df
    data = filtered_df.head(max_incidents).to_dict(orient="records")

    style = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }
      .report-container { max-width: 900px; margin: 2em auto; }
      .incident-details { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 1em; }
      summary { cursor: pointer; padding: 1em; font-size: 1.1em; list-style: none; } /* Hide default arrow */
      summary::-webkit-details-marker { display: none; } /* Hide arrow for Chrome/Safari */
      summary::before { content: '► '; margin-right: 0.5em; } /* Custom closed arrow */
      .incident-details[open] > summary::before { content: '▼ '; } /* Custom open arrow */
      summary:hover { background: #1f242c; }
      .incident-id { font-weight: bold; color: #58a6ff; }
      .short-desc { color: #8b949e; margin-left: 1em; }
      .incident-content { padding: 0 1.5em 1.5em 1.5em; }
      h3 { border-bottom: 1px solid #30363d; padding-bottom: 0.5em; color: #f0f6fc; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }
      td { padding: 0.6em; }
      .field-name { font-weight: bold; color: #8b949e; width: 150px; }
      .work-notes-container { display: flex; flex-direction: column; gap: 1em; }
      .work-note-entry { background-color: #010409; border: 1px solid #30363d; border-radius: 6px; }
      .timestamp { font-weight: bold; color: #8b949e; padding: 0.5em 1em; background-color: #1f242c; border-bottom: 1px solid #30363d; font-family: monospace; }
      .note-text { padding: 1em; margin: 0; white-space: pre-wrap; word-wrap: break-word; }
      pre.react-trace { background: #010409; padding: 1em; border-radius: 6px; white-space: pre-wrap; word-wrap: break-word; font-family: "SF Mono", "Consolas", "Liberation Mono", Menlo, monospace; font-size: 0.9em; }
    #   .thought { color: #58a6ff; }
    #   .action { color: #56d364; }
    #   .obs { color: #e3b341; }
    </style>
    """

    html_doc = f"<!DOCTYPE html><html><head><title>Incident Report</title>{style}</head><body>"
    html_doc += "<div class='report-container'><h1>Incident Analysis Report</h1>"

    for inc_data in data:
        html_doc += render(inc_data)

    html_doc += "</div></body></html>"

    Path(output_file).write_text(html_doc, encoding="utf-8")
    print(f"Wrote {len(data)} incidents to {output_file}")


if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    if len(sys.argv) > 3:
        max_incidents = int(sys.argv[3])
    else:
        max_incidents = 15
    if len(sys.argv) > 4:
        selected_criteria = sys.argv[4]
    else:
        selected_criteria = None
    main(input_filename, output_filename, max_incidents, selected_criteria)
