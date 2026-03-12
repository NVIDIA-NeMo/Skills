#!/bin/bash

# --- CONFIGURATION ---
BASE_PATH="/lustre/fsw/portfolios/llmservice/users/wasiuddina/swe-bench/repos"
REPOS_FILE="repos.txt"
# Number of concurrent clones (Adjust based on your bandwidth/CPU)
CONCURRENT_JOBS=10

# 1. Check if the repos.txt file exists
if [ ! -f "$REPOS_FILE" ]; then
    echo "Error: $REPOS_FILE not found."
    exit 1
fi

# 2. Define the worker function
do_mirror() {
    local REPO_INPUT=$(echo "$1" | xargs)
    local BASE_PATH="$2"

    # Skip empty or commented lines
    [[ -z "$REPO_INPUT" || "$REPO_INPUT" =~ ^# ]] && return

    local OWNER="${REPO_INPUT%/*}"
    local REPO="${REPO_INPUT#*/}"
    local TARGET_DIR="$BASE_PATH/$OWNER/$REPO"
    local GIT_URL="https://github.com/${REPO_INPUT}.git"

    mkdir -p "$BASE_PATH/$OWNER"

    if [ -d "$TARGET_DIR" ]; then
        echo "Updating: $REPO_INPUT"
        cd "$TARGET_DIR" && git remote update && cd - > /dev/null
    else
        echo "Mirroring: $REPO_INPUT"
        if git clone --mirror "$GIT_URL" "$TARGET_DIR"; then
            echo "Success: $REPO_INPUT"
        else
            echo "Error: $REPO_INPUT"
            rm -rf "$TARGET_DIR"
        fi
    fi
}

# Export the function and variables so parallel can see them
export -f do_mirror

# 3. Run in parallel
# --jobs: number of simultaneous clones
# --eta: shows estimated time remaining
parallel --jobs $CONCURRENT_JOBS --eta do_mirror {} "$BASE_PATH" < "$REPOS_FILE"

echo "----------------------------------------------------"
echo "Done processing all repositories."
