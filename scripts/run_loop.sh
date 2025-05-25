#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run continuously until manually stopped
while true; do
    # Run the run_shuffle_kill script using the full path
    "$SCRIPT_DIR/run_shuffle_kill.sh"
    
    # Optional: Add a small delay between iterations (e.g., 1 second)
    sleep 1
done
