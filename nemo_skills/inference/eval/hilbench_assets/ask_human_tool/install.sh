#!/usr/bin/env bash

# install.sh for ask_human tool (Client)

echo "🔧 Installing ask_human tool (client)..."

# Get the directory of this script
bundle_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Make the binary executable
chmod +x "$bundle_dir/bin/ask_human" || {
    echo "❌ ERROR: Failed to make ask_human executable"
    exit 1
}

echo "✅ ask_human tool installed successfully!"
echo "   - Binary: $bundle_dir/bin/ask_human"
