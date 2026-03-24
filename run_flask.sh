#!/bin/bash
# =============================================================================
# Run the LongMemEval ↔ Hindsight Flask Bridge with Azure OpenAI
# =============================================================================
#
# Usage:
#   bash run_flask.sh
#
# Prerequisites:
#   1. az login --tenant <your-tenant-id>
#   2. cp .env.azure.example .env && edit .env with your values
#   3. uv sync --package hindsight-dev (first time only)
#
# NOTE: Shell-level config (PATH, etc.) goes here, NOT in .env.
#       python-dotenv loads .env but treats values as literals — it does NOT
#       expand $PATH, $HOME, or other shell variables.
# =============================================================================

set -euo pipefail

# --- WSL PATH fix ---
# Azure CLI is installed on the Windows side; add it to Linux PATH so that
# azure-identity's AzureCliCredential can find and invoke z.
AZ_CLI_WIN="/mnt/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin"
if [ -d "$AZ_CLI_WIN" ]; then
    export PATH="$AZ_CLI_WIN:$PATH"
fi

# Verify az is reachable
if ! command -v az &>/dev/null; then
    echo "ERROR: az CLI not found. Install Azure CLI or fix PATH." >&2
    echo "  Windows: https://aka.ms/installazurecli" >&2
    echo "  WSL:     curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash" >&2
    exit 1
fi

# Verify logged in
if ! az account show &>/dev/null 2>&1; then
    echo "ERROR: Not logged in to Azure CLI. Run:" >&2
    echo "  az login --tenant <your-tenant-id>" >&2
    exit 1
fi

cd "$(dirname "$0")"
# If this script is NOT inside the repo, cd to the repo
if [ ! -f "hindsight-dev/benchmarks/longmemeval/flask_app.py" ]; then
    cd /home/$USER/repos/hindsight
fi

echo "Starting LongMemEval Flask Bridge..."
echo "  az account: $(az account show --query name -o tsv 2>/dev/null || echo 'unknown')"
echo "  Working dir: $(pwd)"

exec uv run --package hindsight-dev python hindsight-dev/benchmarks/longmemeval/flask_app.py