#!/usr/bin/env bash
# Simple setup script for Hybrid-Robot-Navigation-System
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install system dependencies if apt-get is available
if command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-pip libgl1-mesa-glx libegl1-mesa
fi

# Install Python packages
python3 -m pip install --upgrade pip
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"
