#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/igitman/workspace/NeMo-Skills-aceproof-mp}"
SESSION_LABEL="${SESSION_LABEL:-repeat10-mp}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
STATE_DIR="${STATE_DIR:-/tmp/repeat10_endpoint_failure_monitor}"
NOTIFY="${NOTIFY:-/home/igitman/.claude/bin/notify-slack}"
LOG_GLOB_ROOT="${LOG_GLOB_ROOT:-$ROOT/outputs}"
STATE_FILE="$STATE_DIR/seen.tsv"
STATUS_LOG="$STATE_DIR/status.log"
PATTERN='Internal Server Error|nginx/1\.27\.5|litellm\.InternalServerError|Transient OpenAI-compatible endpoint error|HTTP Error 500'

mkdir -p "$STATE_DIR"
touch "$STATE_FILE" "$STATUS_LOG"

notify_once() {
  local key="$1"
  local message="$2"
  if grep -Fqx "$key" "$STATE_FILE"; then
    return 0
  fi
  printf '%s\n' "$key" >>"$STATE_FILE"
  "$NOTIFY" "$message"
}

while true; do
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  raw_scan="$STATE_DIR/latest_scan.raw.txt"
  {
    echo "[$now] scanning recent logs under $LOG_GLOB_ROOT"
    find "$LOG_GLOB_ROOT" -type f -name '*.log' -mmin -3 -print0 2>/dev/null \
      | xargs -0 -r rg -n "$PATTERN" || true
  } >"$raw_scan"

  cutoff_epoch="$(date -d '5 minutes ago' '+%s')"
  {
    head -n 1 "$raw_scan"
    awk -v cutoff="$cutoff_epoch" '
      match($0, /20[0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]/) {
        ts = substr($0, RSTART, RLENGTH)
        gsub(/[-:]/, " ", ts)
        if (mktime(ts) >= cutoff) print
      }
    ' "$raw_scan"
  } >"$STATE_DIR/latest_scan.txt"

  if [[ -s "$STATE_DIR/latest_scan.txt" ]] && rg -q "$PATTERN" "$STATE_DIR/latest_scan.txt"; then
    sig="$(rg "$PATTERN" "$STATE_DIR/latest_scan.txt" | tail -n 5)"
    if rg -q 'nginx/1\.27\.5|Internal Server Error|HTTP Error 500' <<<"$sig"; then
      family="nginx_500"
    elif rg -q 'job .* not found|queue_job_failed' <<<"$sig"; then
      family="queue_job_failed"
    elif rg -q 'Connection error|Transient OpenAI-compatible endpoint error' <<<"$sig"; then
      family="connection_error"
    else
      family="other_endpoint_error"
    fi
    # Coalesce noisy retry storms while still allowing a later outage to notify.
    bucket="$(date '+%Y%m%d%H')-$((10#$(date '+%M') / 30))"
    key="$SESSION_LABEL:$family:$bucket"
    notify_once "$key" "AceProof repeat10 endpoint failure detected at $now in $SESSION_LABEL ($family).

Recent log signature:
\`\`\`text
$sig
\`\`\`

I will keep monitoring and will avoid launching more retry pressure if this keeps recurring."
  fi

  sleep "$INTERVAL_SECONDS"
done
