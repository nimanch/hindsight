# Azure Foundry OpenAI + LongMemEval Benchmark Guide

This guide covers setting up Hindsight with Azure AI Foundry (Azure OpenAI) endpoints and running the LongMemEval benchmark via the Flask bridge application.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Azure Setup](#azure-setup)
  - [1. Azure AI Foundry Resource](#1-azure-ai-foundry-resource)
  - [2. Deploy a Model](#2-deploy-a-model)
  - [3. Authentication](#3-authentication)
- [Hindsight Server Setup](#hindsight-server-setup)
  - [1. Clone and Install](#1-clone-and-install)
  - [2. Configure Environment](#2-configure-environment)
  - [3. Start the Flask Bridge](#3-start-the-flask-bridge)
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
  - [Category Test (10 items)](#category-test-10-items)
  - [Full Benchmark (500 items)](#full-benchmark-500-items)
- [Changing Azure Endpoints](#changing-azure-endpoints)
- [Environment Variables Reference](#environment-variables-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **WSL (Windows Subsystem for Linux)** — Hindsight requires Linux; it does not run well on native Windows
- **Python 3.11+**
- **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Azure CLI** (`az`) — for Entra ID authentication
- An **Azure AI Foundry** or **Azure OpenAI** resource with a deployed model

---

## Azure Setup

### 1. Azure AI Foundry Resource

You need an Azure AI Foundry (or Azure OpenAI) resource. Your endpoint URL will look like:

```
https://<your-resource-name>.cognitiveservices.azure.com/
```

or for older Azure OpenAI resources:

```
https://<your-resource-name>.openai.azure.com/
```

To find your endpoint:
```bash
az cognitiveservices account list \
  --query "[].{name:name, endpoint:properties.endpoint}" \
  -o table
```

### 2. Deploy a Model

Deploy a model (e.g., `gpt-4o`) in your Azure AI Foundry resource. The **deployment name** is what Hindsight uses to route API calls.

- If your deployment name matches the model name (e.g., both are `gpt-4o`), no extra config is needed
- If they differ, set `HINDSIGHT_API_LLM_AZURE_DEPLOYMENT_NAME` to your deployment name

**Important:** Azure gpt-4o deployments typically support a maximum of **16,384 completion tokens** (vs 64,000 on OpenAI). Hindsight's Azure provider automatically caps `max_completion_tokens` to 16,384. Override with `HINDSIGHT_API_LLM_AZURE_MAX_COMPLETION_TOKENS` if your deployment supports more.

### 3. Authentication

Two authentication methods are supported:

#### Option A: Entra ID (Recommended)

Uses `DefaultAzureCredential` from the Azure Identity SDK. This is the default and recommended method for enterprise use.

```bash
# Login to Azure CLI with the correct tenant
az login --tenant <your-tenant-id>

# Set the subscription containing your resource
az account set -s "<subscription-name>"

# Verify you can get a token for cognitive services
az account get-access-token --resource https://cognitiveservices.azure.com --query tenant -o tsv
```

**Cross-tenant note:** If your Azure CLI is logged into a different tenant than your resource, set:
```bash
export AZURE_TENANT_ID=<tenant-id-of-your-resource>
```

Ensure the `az` CLI is on your PATH inside the Python environment. In WSL, you may need:
```bash
export PATH="$PATH:/mnt/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin"
```

#### Option B: API Key

```bash
export HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID=false
export HINDSIGHT_API_LLM_API_KEY=<your-azure-api-key>
```

Find your API key:
```bash
az cognitiveservices account keys list \
  --name <resource-name> \
  --resource-group <resource-group> \
  --query key1 -o tsv
```

---

## Hindsight Server Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/vectorize-io/hindsight.git
cd hindsight

# Checkout the Azure feature branch
git checkout feature/azure-foundry-openai

# Install dependencies (uv manages the Python workspace)
uv sync --package hindsight-dev
```

### 2. Configure Environment

Copy the example environment file and edit it:

```bash
cp .env.azure.example .env
```

Edit `.env` with your values:

```bash
# Required: Azure provider settings
export HINDSIGHT_API_LLM_PROVIDER=azure
export HINDSIGHT_API_LLM_MODEL=gpt-4o
export HINDSIGHT_API_LLM_AZURE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Authentication (Entra ID is default)
export HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID=true

# Cross-tenant auth (if needed)
export AZURE_TENANT_ID=<tenant-id-of-your-resource>

# Azure gpt-4o max completion tokens (default: 16384)
export HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=16000

# Database: pg0 = embedded PostgreSQL (zero setup)
export HINDSIGHT_API_DATABASE_URL=pg0

# Flask port
export FLASK_PORT=5001

# Ensure az CLI is on PATH (WSL)
export PATH="$PATH:/mnt/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin"
```

### 3. Start the Flask Bridge

```bash
# Source your environment
source .env

# Start the Flask bridge server
uv run --package hindsight-dev python hindsight-dev/benchmarks/longmemeval/flask_app.py
```

You should see:

```
╔══════════════════════════════════════════════════════════════╗
║        LongMemEval ↔ Hindsight Flask Bridge                 ║
╠══════════════════════════════════════════════════════════════╣
║  Provider : azure                                           ║
║  Model    : gpt-4o                                          ║
║  Azure EP : https://your-resource.cognitiveservices.azure.com/║
║  Entra ID : true                                            ║
║  Port     : 5001                                            ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Flask Bridge API Reference

The Flask bridge exposes 8 endpoints that wrap Hindsight's memory engine for LongMemEval benchmarking.

### POST /api/init

Initialize the Hindsight memory engine. **Must be called before any other operation.**

Starts the embedded PostgreSQL database (pg0), loads the LongMemEval dataset, and verifies Azure OpenAI connectivity.

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

Health check with LLM connectivity status.

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

Statistics about the loaded LongMemEval dataset.

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
  "sample_item_id": "e47becba",
  "sample_num_sessions": 53,
  "ingested_banks": 0
}
```

---

### GET /api/dataset/item/\<id\>

Details for a specific dataset item, including its QA pairs.

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
      "category": "single-session-user",
      "question_date": "Tue, 30 May 2023 00:00:00 GMT"
    }
  ],
  "bank_id": "longmemeval_e47becba",
  "is_ingested": false
}
```

---

### POST /api/index

Ingest LongMemEval sessions into Hindsight memory banks via `retain_batch_async()`.

Each dataset item's conversation sessions are extracted, chunked, and stored as facts in a dedicated memory bank (`longmemeval_<question_id>`).

```bash
# Index all 500 items
curl -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{}'

# Index first 5 items only
curl -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"max_items": 5}'

# Index specific items by question ID
curl -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"question_ids": ["e47becba", "a1b2c3d4"]}'

# Force re-index (clears existing data first)
curl -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"question_ids": ["e47becba"], "force": true}'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_items` | int | all | Maximum dataset items to ingest |
| `question_ids` | list | all | Specific question IDs to ingest |
| `force` | bool | false | Re-ingest even if already done |

**Response:**
```json
{
  "status": "completed",
  "items_processed": 1,
  "items_skipped": 0,
  "total_sessions_ingested": 53,
  "elapsed_seconds": 490.08,
  "details": [
    {
      "item_id": "e47becba",
      "bank_id": "longmemeval_e47becba",
      "sessions_ingested": 53,
      "status": "ok"
    }
  ]
}
```

---

### POST /api/retrieve

Query a memory bank using Hindsight's `recall_async()` — semantic search + reranking + temporal filtering.

```bash
curl -X POST http://localhost:5001/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What degree did I graduate with?",
    "question_id": "e47becba",
    "question_date": "2023-05-30T00:00:00Z",
    "budget": "mid",
    "max_tokens": 8192
  }'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | str | **required** | The query text |
| `question_id` | str | **required** | Dataset item ID (used as bank prefix) |
| `question_date` | str | null | ISO-8601 date for temporal context |
| `budget` | str | "mid" | Search depth: `low`, `mid`, or `high` |
| `max_tokens` | int | 8192 | Maximum tokens to retrieve |

**Response:**
```json
{
  "status": "ok",
  "bank_id": "longmemeval_e47becba",
  "question": "What degree did I graduate with?",
  "num_results": 187,
  "num_entities": 400,
  "num_chunks": 17,
  "elapsed_seconds": 4.49,
  "recall_result": { ... }
}
```

---

### POST /api/answer

Retrieve memories and generate an answer using the LLM (recall + answer generation pipeline).

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
| `question` | str | **required** | The query text |
| `question_id` | str | **required** | Dataset item ID |
| `question_date` | str | null | ISO-8601 date for temporal context |
| `question_type` | str | null | Question category (for prompt tuning) |
| `budget` | str | "mid" | Search depth |
| `max_tokens` | int | 8192 | Max recall tokens |
| `context_format` | str | "json" | `json` or `structured` |

**Response:**
```json
{
  "status": "ok",
  "answer": "Business Administration",
  "reasoning": "According to the retrieved context, the user graduated with a degree in Business Administration...",
  "recall_time_seconds": 3.14,
  "generation_time_seconds": 8.87,
  "num_results": 186,
  "num_entities": 401
}
```

---

### POST /api/benchmark/run

Run the full LongMemEval benchmark end-to-end (index → recall → answer → judge). This calls the existing `run_benchmark()` function from `longmemeval_benchmark.py`.

```bash
# Run full benchmark
curl -X POST http://localhost:5001/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{
    "max_items": 10,
    "thinking_budget": 500,
    "max_tokens": 8192,
    "context_format": "json"
  }'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_items` | int | all | Max items to evaluate |
| `max_questions_per_item` | int | all | Max questions per item |
| `thinking_budget` | int | 500 | Search budget for spreading activation |
| `max_tokens` | int | 8192 | Max recall tokens |
| `skip_ingestion` | bool | false | Skip indexing (use existing data) |
| `context_format` | str | "json" | `json` or `structured` |
| `results_filename` | str | benchmark_results.json | Output filename |
| `max_concurrent_items` | int | 1 | Parallel item processing |
| `category` | str | null | Filter by question type |

**Response:**
```json
{
  "status": "completed",
  "summary": {
    "total_items": 10,
    "overall_metrics": {
      "accuracy": 72.5,
      "total": 10,
      "correct": 7,
      "invalid": 1
    },
    "elapsed_seconds": 1234.56
  },
  "full_results": { ... }
}
```

---

## Running E2E Tests

### Quick Smoke Test (1 item)

Tests the full pipeline: init → index → retrieve → answer. Takes ~10 minutes.

```bash
# 1. Start the server
source .env
uv run --package hindsight-dev python hindsight-dev/benchmarks/longmemeval/flask_app.py &

# 2. Initialize
curl -s -X POST http://localhost:5001/api/init | python3 -m json.tool

# 3. Index 1 item (53 sessions, ~8 min via Azure gpt-4o)
curl -s -X POST http://localhost:5001/api/index \
  -H "Content-Type: application/json" \
  -d '{"question_ids": ["e47becba"]}' | python3 -m json.tool

# 4. Retrieve memories
curl -s -X POST http://localhost:5001/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "What degree did I graduate with?", "question_id": "e47becba"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Results: {d[\"num_results\"]}, Entities: {d[\"num_entities\"]}')"

# 5. Generate answer (expected: "Business Administration")
curl -s -X POST http://localhost:5001/api/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What degree did I graduate with?", "question_id": "e47becba", "question_date": "2023-05-30T00:00:00Z"}' \
  | python3 -m json.tool
```

**Expected output for step 5:**
```json
{
  "answer": "Business Administration",
  "recall_time_seconds": 3.14,
  "generation_time_seconds": 8.87,
  "num_results": 186
}
```

### Category Test (10 items)

Run 10 items from one category with full benchmark evaluation:

```bash
curl -s -X POST http://localhost:5001/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{
    "max_items": 10,
    "category": "single-session-user",
    "context_format": "json"
  }' | python3 -m json.tool
```

### Full Benchmark (500 items)

Run the complete LongMemEval benchmark. This will take several hours:

```bash
curl -s -X POST http://localhost:5001/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{
    "thinking_budget": 500,
    "max_tokens": 8192,
    "max_concurrent_items": 1,
    "context_format": "json"
  }' | python3 -m json.tool
```

Results are saved to `hindsight-dev/benchmarks/longmemeval/results/benchmark_results.json`.

**Re-running failed questions only:**
```bash
# After an initial run, re-test only the questions that failed
curl -s -X POST http://localhost:5001/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"skip_ingestion": true, "only_failed": true}' | python3 -m json.tool
```

---

## Changing Azure Endpoints

To switch to a different Azure OpenAI resource:

### 1. Update the endpoint

```bash
# Edit .env or export directly:
export HINDSIGHT_API_LLM_AZURE_ENDPOINT=https://new-resource.cognitiveservices.azure.com/
```

### 2. Change the model / deployment

```bash
# If deploying a different model:
export HINDSIGHT_API_LLM_MODEL=gpt-4o-mini
export HINDSIGHT_API_LLM_AZURE_DEPLOYMENT_NAME=my-gpt4o-mini-deployment

# Adjust completion token limit for the new model
export HINDSIGHT_API_LLM_AZURE_MAX_COMPLETION_TOKENS=16384
export HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=16000
```

### 3. Switch tenants

```bash
# If the new resource is in a different tenant:
az login --tenant <new-tenant-id>
az account set -s "<new-subscription>"
export AZURE_TENANT_ID=<new-tenant-id>
```

### 4. Switch to API key auth

```bash
export HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID=false
export HINDSIGHT_API_LLM_API_KEY=<new-api-key>
```

### 5. Restart the Flask server

```bash
# Stop the old server (Ctrl+C), then:
source .env
uv run --package hindsight-dev python hindsight-dev/benchmarks/longmemeval/flask_app.py
```

### 6. Re-initialize

```bash
curl -X POST http://localhost:5001/api/init
```

The embedded pg0 database persists data across restarts. Previously ingested data will still be available. Use `"force": true` in `/api/index` to re-ingest with the new model.

---

## Environment Variables Reference

### Azure Provider (Required)

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_API_LLM_PROVIDER` | `groq` | Set to `azure` for Azure Foundry |
| `HINDSIGHT_API_LLM_MODEL` | `gpt-4o` | Model name (also used as deployment name) |
| `HINDSIGHT_API_LLM_AZURE_ENDPOINT` | — | Azure resource endpoint URL |
| `HINDSIGHT_API_LLM_AZURE_API_VERSION` | `2024-12-01-preview` | Azure OpenAI API version |
| `HINDSIGHT_API_LLM_AZURE_DEPLOYMENT_NAME` | (model name) | Override deployment name if different from model |
| `HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID` | `true` | Use Entra ID auth (`true`) or API key (`false`) |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_API_LLM_API_KEY` | — | API key (required when `ENTRA_ID=false`) |
| `AZURE_TENANT_ID` | — | Force token from specific tenant (cross-tenant auth) |

### Token Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS` | `64000` | Max tokens for fact extraction. Set to `16000` for Azure gpt-4o |
| `HINDSIGHT_API_LLM_AZURE_MAX_COMPLETION_TOKENS` | `16384` | Cap applied to all Azure LLM calls |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_API_DATABASE_URL` | `pg0` | Database URL. `pg0` = embedded PostgreSQL |

### Flask

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | `5000` | Flask server port |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

---

## Troubleshooting

### "Tenant provided in token does not match resource token"

Your Azure CLI token is from a different tenant than the resource.

```bash
# Find which tenant your resource is in
az cognitiveservices account list -o table

# Login to the correct tenant
az login --tenant <resource-tenant-id>
az account set -s "<subscription-name>"

# Or set the tenant ID env var
export AZURE_TENANT_ID=<resource-tenant-id>
```

### "Azure CLI not found on path"

`DefaultAzureCredential` can't find `az`. Ensure it's on your PATH:

```bash
# WSL: add Windows az CLI to PATH
export PATH="$PATH:/mnt/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin"

# Or install az CLI natively in WSL
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### "max_tokens is too large"

Your Azure deployment has a lower token limit than expected.

```bash
# Cap all Azure LLM calls (default: 16384)
export HINDSIGHT_API_LLM_AZURE_MAX_COMPLETION_TOKENS=16384

# Cap the retain (fact extraction) operation specifically
export HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=16000
```

### "Invalid LLM provider: azure"

You're on an older version of the code. Ensure you're on the `feature/azure-foundry-openai` branch:

```bash
git checkout feature/azure-foundry-openai
uv sync --package hindsight-dev
```

### Slow indexing

Each session is processed through the LLM for fact extraction. Tips:
- Use a faster model deployment (e.g., `gpt-4o-mini`)
- Index fewer items for testing: `{"max_items": 5}`
- Data persists in pg0 — use `"skip_ingestion": true` to skip re-indexing

### pg0 database issues

The embedded PostgreSQL stores data in `~/.pg0/`. To reset:

```bash
rm -rf ~/.pg0/instances/hindsight*
```