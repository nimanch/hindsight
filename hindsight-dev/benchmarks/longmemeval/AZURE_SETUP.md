# Azure Foundry OpenAI + Hindsight: Setup, API & E2E Testing Guide

A complete guide to building Hindsight from source, configuring it with Azure AI Foundry (Azure OpenAI) endpoints, and running the LongMemEval benchmark via the Flask bridge application.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Build from Source](#build-from-source)
  - [1. Install System Dependencies](#1-install-system-dependencies)
  - [2. Clone the Repository](#2-clone-the-repository)
  - [3. Install Python Dependencies](#3-install-python-dependencies)
  - [4. Verify the Build](#4-verify-the-build)
- [Azure Setup](#azure-setup)
  - [1. Azure AI Foundry Resource](#1-azure-ai-foundry-resource)
  - [2. Deploy a Model](#2-deploy-a-model)
  - [3. Authentication](#3-authentication)
- [Configure and Run](#configure-and-run)
  - [1. Create Environment File](#1-create-environment-file)
  - [2. Start the Flask Bridge](#2-start-the-flask-bridge)
- [Flask Bridge API Reference](#flask-bridge-api-reference)
  - [POST /api/init](#post-apiinit)
  - [GET /api/health](#get-apihealth)
  - [GET /api/dataset/info](#get-apidatasetinfo)
  - [GET /api/dataset/item/\<id\>](#get-apidatasetitemid)
  - [POST /api/index](#post-apiindex)
  - [POST /api/retrieve](#post-apiretrieve)
  - [POST /api/answer](#post-apianswer)
  - [POST /api/benchmark/run](#post-apibenchmarkrun)
- [Running E2E Tests](#running-e2e-tests)
  - [Quick Smoke Test (1 item)](#quick-smoke-test-1-item)
  - [Category Test](#category-test)
  - [Full Benchmark (500 items)](#full-benchmark-500-items)
  - [Using the Built-in Benchmark Script](#using-the-built-in-benchmark-script)
- [Changing Azure Endpoints](#changing-azure-endpoints)
- [Environment Variables Reference](#environment-variables-reference)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **WSL 2** (Ubuntu) | Ubuntu 22.04+ | Hindsight uses embedded PostgreSQL (pg0) which requires Linux |
| **Python** | 3.11+ | Required by Hindsight |
| **uv** | latest | Python package/workspace manager from Astral |
| **Azure CLI** | 2.60+ | For Entra ID authentication |
| **Git** | 2.25+ | For cloning the repository |
| **curl** | any | For downloading the LongMemEval dataset |

### Windows Users

All commands must be run **inside WSL**. Open a WSL terminal:

```powershell
# From PowerShell / Windows Terminal
wsl
```

The repo will live in WSL's filesystem (e.g., `/home/<user>/repos/hindsight`).
To browse from Windows Explorer, navigate to: `\\wsl$\Ubuntu\home\<user>\repos\hindsight`

---

## Build from Source

### 1. Install System Dependencies

```bash
# Update package lists
sudo apt update

# Python 3.11+ (Ubuntu 22.04 ships 3.10, you may need deadsnakes PPA)
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Build tools (needed for some Python packages)
sudo apt install -y build-essential libffi-dev libssl-dev

# Git
sudo apt install -y git curl
```

### 2. Clone the Repository

```bash
# Create a workspace directory
mkdir -p ~/repos && cd ~/repos

# Clone from GitHub
git clone https://github.com/vectorize-io/hindsight.git
cd hindsight

# Switch to the Azure feature branch
git checkout feature/azure-foundry-openai
```

### 3. Install Python Dependencies

Hindsight uses [uv](https://docs.astral.sh/uv/) as its workspace manager. Install it first:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (or restart shell)
export PATH="$HOME/.local/bin:$PATH"
```

Now install all dependencies. The workspace has multiple packages — you need `hindsight-dev` which pulls in everything:

```bash
# Install all workspace packages and their dependencies
# This downloads Python packages, builds native extensions, and sets up the venv
uv sync --package hindsight-dev

# This installs:
#   hindsight-api-slim  - Core API (FastAPI, LLM providers, memory engine)
#   hindsight-all       - Server, embedded client (HindsightEmbedded)
#   hindsight-dev       - Dev tools, benchmarks, Flask bridge
#   hindsight-embed     - Embedded PostgreSQL (pg0) daemon
#   + azure-identity    - Azure Entra ID authentication
#   + flask             - Flask web framework for the bridge app
```

**Note:** The first `uv sync` may take a few minutes. It downloads PyTorch (CPU), sentence-transformers, and other ML dependencies (~2GB total).

### 4. Verify the Build

```bash
# Verify core imports work
uv run --package hindsight-dev python -c "
from hindsight_api.engine.providers.azure_openai_llm import AzureOpenAILLM
from hindsight_api.config import HindsightConfig
from benchmarks.longmemeval.flask_app import create_app
print('Build verified successfully!')
print(f'  AzureOpenAILLM: OK')
print(f'  HindsightConfig: OK')
print(f'  Flask app: OK')
app = create_app()
routes = [r.rule for r in app.url_map.iter_rules() if not r.rule.startswith('/static')]
print(f'  Routes: {len(routes)} endpoints registered')
"
```

Expected output:
```
Build verified successfully!
  AzureOpenAILLM: OK
  HindsightConfig: OK
  Flask app: OK
  Routes: 8 endpoints registered
```

---

## Azure Setup

### 1. Azure AI Foundry Resource

You need an Azure AI Foundry (or Azure OpenAI) resource. Your endpoint URL will look like:

```
https://<your-resource-name>.cognitiveservices.azure.com/
```

To find your endpoint:
```bash
# List all cognitive services resources
az cognitiveservices account list \
  --query "[].{name:name, endpoint:properties.endpoint}" \
  -o table
```

### 2. Deploy a Model

Deploy a model (e.g., `gpt-4o`) in your Azure AI Foundry resource:

1. Go to [Azure AI Foundry](https://ai.azure.com) or [Azure Portal](https://portal.azure.com)
2. Navigate to your resource → **Model deployments** → **Deploy model**
3. Choose `gpt-4o` (or your preferred model)
4. Note the **deployment name** (defaults to the model name)

**Important:** Azure gpt-4o deployments typically support max **16,384 completion tokens** (vs 64,000 on OpenAI direct). The Azure provider auto-caps this, but you must also set `HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=16000`.

### 3. Authentication

#### Option A: Entra ID (Recommended)

Uses `DefaultAzureCredential` — works with `az login`, managed identities, and service principals.

```bash
# Login to Azure CLI
az login --tenant <your-tenant-id>

# Set your subscription
az account set -s "<subscription-name>"

# Verify token works for cognitive services
az account get-access-token \
  --resource https://cognitiveservices.azure.com \
  --query tenant -o tsv
```

**Cross-tenant note:** If your resource is in a different Azure AD tenant, set:
```bash
export AZURE_TENANT_ID=<tenant-id-where-resource-lives>
```

#### Option B: API Key

```bash
export HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID=false
export HINDSIGHT_API_LLM_API_KEY=<your-api-key>
```

Get your key:
```bash
az cognitiveservices account keys list \
  --name <resource-name> \
  --resource-group <resource-group> \
  --query key1 -o tsv
```

---

## Configure and Run

### 1. Create Environment File

```bash
cd ~/repos/hindsight

# Copy the template
cp .env.azure.example .env

# Edit with your values
vi .env   # or: code .env
```

**Minimum required `.env` for Azure:**

```bash
# Provider
export HINDSIGHT_API_LLM_PROVIDER=azure
export HINDSIGHT_API_LLM_MODEL=gpt-4o

# Azure endpoint
export HINDSIGHT_API_LLM_AZURE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Auth (Entra ID)
export HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID=true
# export AZURE_TENANT_ID=<if-cross-tenant>

# Token limits (Azure gpt-4o = 16384 max)
export HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=16000

# Database (embedded, zero setup)
export HINDSIGHT_API_DATABASE_URL=pg0

# Flask
export FLASK_PORT=5001

# WSL: ensure az CLI is on PATH
export PATH="$PATH:/mnt/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin"
```

### 2. Start the Flask Bridge

```bash
# Load environment
source .env

# Start the server
uv run --package hindsight-dev python hindsight-dev/benchmarks/longmemeval/flask_app.py
```

You should see:
```
╔══════════════════════════════════════════════════════════════╗
║        LongMemEval ↔ Hindsight Flask Bridge                 ║
║  Provider : azure                                           ║
║  Model    : gpt-4o                                          ║
║  Port     : 5001                                            ║
╚══════════════════════════════════════════════════════════════╝
 * Running on http://127.0.0.1:5001
```

---

## Flask Bridge API Reference

All endpoints accept and return JSON. Base URL: `http://localhost:5001`

### POST /api/init

**Initialize the memory engine.** Must be called once before any other operation.

Starts pg0 (embedded PostgreSQL), creates the LLM provider, downloads and loads the LongMemEval dataset (500 items), and verifies Azure OpenAI connectivity.

```bash
curl -X POST http://localhost:5001/api/init \
  -H "Content-Type: application/json"
```

**Response:**
```json
{
  "status": "initialized",
  "provider": "azure",
  "model": "gpt-4o",
  "dataset_items": 500
}
```

---

### GET /api/health

**Health check** with provider info and dataset status.

```bash
curl http://localhost:5001/api/health
```

**Response:**
```json
{
  "status": "ok",
  "provider": "azure",
  "model": "gpt-4o",
  "azure_endpoint": "https://your-resource.cognitiveservices.azure.com/",
  "dataset_loaded": true,
  "dataset_items": 500,
  "ingested_banks": 3
}
```

---

### GET /api/dataset/info

**Dataset statistics** — item counts by category.

```bash
curl http://localhost:5001/api/dataset/info
```

**Response:**
```json
{
  "total_items": 500,
  "categories": {
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
    "multi-session": 133,
    "temporal-reasoning": 133,
    "knowledge-update": 78
  },
  "ingested_banks": 0
}
```

---

### GET /api/dataset/item/\<id\>

**Single item details** including QA pairs and session metadata.

```bash
curl http://localhost:5001/api/dataset/item/e47becba
```

**Response:**
```json
{
  "question_id": "e47becba",
  "question_type": "single-session-user",
  "num_sessions": 53,
  "num_qa_pairs": 1,
  "qa_pairs": [
    {
      "question": "What degree did I graduate with?",
      "answer": "Business Administration",
      "category": "single-session-user"
    }
  ],
  "bank_id": "longmemeval_e47becba",
  "is_ingested": false
}
```

---

### POST /api/index

**Ingest LongMemEval sessions** into Hindsight memory banks. Each item's conversation sessions are processed through the LLM for fact extraction and stored in a dedicated bank.

```bash
# Index specific items
curl -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"question_ids": ["e47becba"]}'

# Index first N items
curl -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"max_items": 5}'

# Force re-index (clears existing data)
curl -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"question_ids": ["e47becba"], "force": true}'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_items` | int | all | Maximum items to ingest |
| `question_ids` | list[str] | all | Specific question IDs to ingest |
| `force` | bool | false | Re-ingest even if already done |

**Response:**
```json
{
  "status": "completed",
  "items_processed": 1,
  "total_sessions_ingested": 53,
  "elapsed_seconds": 490.08,
  "details": [
    { "item_id": "e47becba", "sessions_ingested": 53, "status": "ok" }
  ]
}
```

**Note:** Indexing is the slowest step (~8 min per item with Azure gpt-4o). Data persists in pg0 across restarts — use `"skip_ingestion": true` in `/api/benchmark/run` to reuse indexed data.

---

### POST /api/retrieve

**Query memories** using Hindsight's recall engine — semantic search, graph expansion, reranking, and temporal filtering.

```bash
curl -X POST http://localhost:5001/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What degree did I graduate with?",
    "question_id": "e47becba",
    "question_date": "2023-05-30T00:00:00Z"
  }'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | str | **required** | Query text |
| `question_id` | str | **required** | Item ID (bank prefix) |
| `question_date` | str | null | ISO-8601 date for temporal context |
| `budget` | str | `"mid"` | Search depth: `low` / `mid` / `high` |
| `max_tokens` | int | 8192 | Max tokens to retrieve |

**Response:**
```json
{
  "status": "ok",
  "num_results": 187,
  "num_entities": 400,
  "num_chunks": 17,
  "elapsed_seconds": 4.49,
  "recall_result": { "results": [...], "entities": {...}, "chunks": {...} }
}
```

---

### POST /api/answer

**Retrieve + generate answer.** Runs recall, then passes retrieved facts to the LLM for answer generation.

```bash
curl -X POST http://localhost:5001/api/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What degree did I graduate with?",
    "question_id": "e47becba",
    "question_date": "2023-05-30T00:00:00Z",
    "question_type": "single-session-user"
  }'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | str | **required** | Query text |
| `question_id` | str | **required** | Item ID |
| `question_date` | str | null | ISO-8601 date |
| `question_type` | str | null | Category hint |
| `budget` | str | `"mid"` | Search depth |
| `max_tokens` | int | 8192 | Max recall tokens |
| `context_format` | str | `"json"` | `json` or `structured` |

**Response:**
```json
{
  "status": "ok",
  "answer": "Business Administration",
  "reasoning": "According to the retrieved context, the user graduated with...",
  "recall_time_seconds": 3.14,
  "generation_time_seconds": 8.87,
  "num_results": 186
}
```

---

### POST /api/benchmark/run

**Full E2E benchmark** — index → recall → answer → judge (LLM-as-judge evaluation). Calls the existing `run_benchmark()` from `longmemeval_benchmark.py`.

```bash
# Run 10 items
curl -X POST http://localhost:5001/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"max_items": 10, "context_format": "json"}'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_items` | int | all | Max items to evaluate |
| `max_questions_per_item` | int | all | Max questions per item |
| `thinking_budget` | int | 500 | Search budget |
| `max_tokens` | int | 8192 | Max recall tokens |
| `skip_ingestion` | bool | false | Reuse existing indexed data |
| `context_format` | str | `"json"` | `json` or `structured` |
| `results_filename` | str | `benchmark_results.json` | Output file |
| `max_concurrent_items` | int | 1 | Parallel processing |
| `category` | str | null | Filter by question type |

Results are saved to `hindsight-dev/benchmarks/longmemeval/results/<results_filename>`.

---

## Running E2E Tests

### Quick Smoke Test (1 item)

End-to-end test of init → index → retrieve → answer. Takes ~10 minutes.

```bash
# Terminal 1: Start the server
source .env
uv run --package hindsight-dev python hindsight-dev/benchmarks/longmemeval/flask_app.py

# Terminal 2: Run the test sequence
# Step 1: Initialize
curl -s -X POST http://localhost:5001/api/init | python3 -m json.tool

# Step 2: Index 1 item (53 sessions, ~8 min)
curl -s -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"question_ids": ["e47becba"]}' | python3 -m json.tool

# Step 3: Retrieve memories
curl -s -X POST http://localhost:5001/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "What degree did I graduate with?", "question_id": "e47becba"}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Results: {d[\"num_results\"]}, Entities: {d[\"num_entities\"]}')
"

# Step 4: Generate answer (expected: "Business Administration")
curl -s -X POST http://localhost:5001/api/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What degree did I graduate with?",
    "question_id": "e47becba",
    "question_date": "2023-05-30T00:00:00Z"
  }' | python3 -m json.tool
```

**Expected results:**
- Index: 53 sessions ingested
- Retrieve: ~187 results, ~400 entities, ~17 chunks in ~4s
- Answer: `"Business Administration"` (matches ground truth)

### Category Test

Test 10 items from a specific question category:

```bash
curl -s -X POST http://localhost:5001/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{
    "max_items": 10,
    "category": "single-session-user",
    "context_format": "json"
  }' | python3 -m json.tool
```

Available categories: `single-session-user`, `single-session-assistant`, `single-session-preference`, `multi-session`, `temporal-reasoning`, `knowledge-update`

### Full Benchmark (500 items)

Run the complete LongMemEval benchmark. **Estimated time: several hours.**

```bash
curl -s -X POST http://localhost:5001/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{
    "thinking_budget": 500,
    "max_tokens": 8192,
    "max_concurrent_items": 1
  }' | python3 -m json.tool
```

### Using the Built-in Benchmark Script

You can also run the benchmark directly (without the Flask bridge):

```bash
source .env
./scripts/benchmarks/run-longmemeval.sh --max-instances 10
```

Or with Python directly:

```bash
source .env
uv run python hindsight-dev/benchmarks/longmemeval/longmemeval_benchmark.py \
  --max-instances 10 \
  --context-format json
```

---

## Changing Azure Endpoints

### Switch to a different resource

```bash
# 1. Update endpoint
export HINDSIGHT_API_LLM_AZURE_ENDPOINT=https://new-resource.cognitiveservices.azure.com/

# 2. If different tenant, update auth
az login --tenant <new-tenant-id>
export AZURE_TENANT_ID=<new-tenant-id>

# 3. Restart the Flask server and re-init
curl -X POST http://localhost:5001/api/init
```

### Switch model or deployment

```bash
export HINDSIGHT_API_LLM_MODEL=gpt-4o-mini
export HINDSIGHT_API_LLM_AZURE_DEPLOYMENT_NAME=my-custom-deployment

# Adjust token limits for the new model
export HINDSIGHT_API_LLM_AZURE_MAX_COMPLETION_TOKENS=16384
export HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=16000
```

### Switch to API key auth

```bash
export HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID=false
export HINDSIGHT_API_LLM_API_KEY=<your-api-key>
```

### Re-index with new model

Data persists in pg0. Use `"force": true` to re-ingest with the new model:

```bash
curl -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"force": true, "max_items": 5}'
```

---

## Environment Variables Reference

### Core Azure Config

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_API_LLM_PROVIDER` | `groq` | Set to `azure` |
| `HINDSIGHT_API_LLM_MODEL` | `gpt-4o` | Model / deployment name |
| `HINDSIGHT_API_LLM_AZURE_ENDPOINT` | — | Azure resource URL (required) |
| `HINDSIGHT_API_LLM_AZURE_API_VERSION` | `2024-12-01-preview` | API version |
| `HINDSIGHT_API_LLM_AZURE_DEPLOYMENT_NAME` | (model name) | Override if deployment name differs |
| `HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID` | `true` | `true` = Entra ID, `false` = API key |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_API_LLM_API_KEY` | — | Required when `USE_ENTRA_ID=false` |
| `AZURE_TENANT_ID` | — | Force token from specific tenant |

### Token Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS` | `64000` | Set to `16000` for Azure gpt-4o |
| `HINDSIGHT_API_LLM_AZURE_MAX_COMPLETION_TOKENS` | `16384` | Auto-cap for all Azure calls |

### Database & Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_API_DATABASE_URL` | `pg0` | `pg0` = embedded PostgreSQL |
| `FLASK_PORT` | `5000` | Flask server port |
| `FLASK_DEBUG` | `false` | Debug mode |

---

## Project Structure

```
hindsight/
├── .env.azure.example           # Azure config template
├── hindsight-api-slim/          # Core API package
│   └── hindsight_api/
│       ├── config.py            # Configuration & env vars
│       └── engine/
│           ├── memory_engine.py # MemoryEngine (retain, recall, reflect)
│           ├── llm_wrapper.py   # LLM provider factory
│           └── providers/
│               ├── azure_openai_llm.py      # ★ Azure Foundry provider
│               ├── openai_compatible_llm.py # OpenAI/Groq/Ollama
│               ├── anthropic_llm.py         # Anthropic Claude
│               └── gemini_llm.py            # Google Gemini
├── hindsight-all/               # Server + embedded client
│   └── hindsight/
│       ├── server.py            # Background Hindsight server
│       └── embedded.py          # HindsightEmbedded client
├── hindsight-dev/               # Dev tools & benchmarks
│   └── benchmarks/
│       └── longmemeval/
│           ├── AZURE_SETUP.md           # ★ This guide
│           ├── flask_app.py             # ★ Flask bridge (8 endpoints)
│           ├── longmemeval_benchmark.py # Benchmark runner
│           └── datasets/               # Auto-downloaded dataset
└── scripts/
    └── benchmarks/
        └── run-longmemeval.sh   # CLI benchmark runner
```

Files marked with ★ were added by the Azure Foundry feature branch.

---

## Troubleshooting

### "Tenant provided in token does not match resource token"

Your `az login` tenant doesn't match the resource's tenant.

```bash
# Find your resource's tenant
az cognitiveservices account list -o table

# Login to the correct tenant
az login --tenant <resource-tenant-id>
az account set -s "<subscription>"

# Or set env var
export AZURE_TENANT_ID=<resource-tenant-id>
```

### "Azure CLI not found on path"

`DefaultAzureCredential` can't find `az`. In WSL:

```bash
# Option 1: Use Windows az CLI
export PATH="$PATH:/mnt/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin"

# Option 2: Install az CLI natively in WSL
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### "max_tokens is too large: 32768"

Azure deployment has lower token limits. Set:

```bash
export HINDSIGHT_API_LLM_AZURE_MAX_COMPLETION_TOKENS=16384
export HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=16000
```

### "Invalid LLM provider: azure"

Not on the Azure feature branch:

```bash
git checkout feature/azure-foundry-openai
uv sync --package hindsight-dev
```

### Slow indexing

Each session goes through LLM fact extraction. Tips:
- Use `gpt-4o-mini` for faster indexing
- Index fewer items: `{"max_items": 5}`
- Data persists in pg0 — skip re-indexing with `"skip_ingestion": true`

### Reset embedded database

```bash
rm -rf ~/.pg0/instances/hindsight*
```