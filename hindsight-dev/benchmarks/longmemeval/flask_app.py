"""
LongMemEval Flask Bridge Application

A Flask application that bridges the LongMemEval benchmark to Hindsight's
memory engine, enabling E2E testing of indexing and retrieval with Azure
Foundry OpenAI endpoints.

Usage:
    # Set Azure env vars first (see .env.azure.example)
    export HINDSIGHT_API_LLM_PROVIDER=azure
    export HINDSIGHT_API_LLM_MODEL=gpt-4o
    export HINDSIGHT_API_LLM_AZURE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
    export HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID=true

    # Run the Flask app
    python flask_app.py
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

# Add project root to path for imports
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async helper -- run coroutines from synchronous Flask handlers
# ---------------------------------------------------------------------------
_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a dedicated background event loop (created once)."""
    global _loop
    if _loop is None or _loop.is_closed():
        import threading

        _loop = asyncio.new_event_loop()
        t = threading.Thread(target=_loop.run_forever, daemon=True)
        t.start()
    return _loop


def run_async(coro):
    """Schedule *coro* on the background loop and block until done."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # State stored on the app object
    app.extensions["hindsight"] = {
        "memory": None,          # MemoryEngine instance
        "initialized": False,
        "dataset": None,         # LongMemEvalDataset instance
        "dataset_items": None,   # Loaded dataset items
        "ingested_banks": set(), # Track which banks have been ingested
    }

    # ------------------------------------------------------------------
    # Lazy initialisation endpoint
    # ------------------------------------------------------------------
    @app.route("/api/init", methods=["POST"])
    def init_engine():
        """Initialize the Hindsight MemoryEngine.

        Call this once before indexing / retrieval.  Reads LLM config from
        the standard ``HINDSIGHT_API_*`` environment variables.
        """
        state = app.extensions["hindsight"]

        if state["initialized"]:
            return jsonify({"status": "already_initialized"}), 200

        try:
            from benchmarks.common.benchmark_runner import create_memory_engine

            memory = run_async(create_memory_engine())
            state["memory"] = memory
            state["initialized"] = True

            # Preload dataset
            from benchmarks.longmemeval.longmemeval_benchmark import (
                LongMemEvalDataset,
                download_dataset,
            )

            dataset = LongMemEvalDataset()
            dataset_path = Path(__file__).parent / "datasets" / "longmemeval_s_cleaned.json"
            if not dataset_path.exists():
                download_dataset(dataset_path)

            if dataset_path.exists():
                state["dataset"] = dataset
                state["dataset_items"] = dataset.load(dataset_path)
                logger.info(f"Loaded {len(state['dataset_items'])} dataset items")

            return jsonify({
                "status": "initialized",
                "provider": os.getenv("HINDSIGHT_API_LLM_PROVIDER", "unknown"),
                "model": os.getenv("HINDSIGHT_API_LLM_MODEL", "unknown"),
                "dataset_items": len(state["dataset_items"]) if state["dataset_items"] else 0,
            })
        except Exception as e:
            logger.exception("Failed to initialise MemoryEngine")
            return jsonify({"status": "error", "error": str(e), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    @app.route("/api/health", methods=["GET"])
    def health():
        state = app.extensions["hindsight"]
        info: Dict[str, Any] = {
            "status": "ok" if state["initialized"] else "not_initialized",
            "provider": os.getenv("HINDSIGHT_API_LLM_PROVIDER", "not_set"),
            "model": os.getenv("HINDSIGHT_API_LLM_MODEL", "not_set"),
            "azure_endpoint": os.getenv("HINDSIGHT_API_LLM_AZURE_ENDPOINT", "not_set"),
            "dataset_loaded": state["dataset_items"] is not None,
            "dataset_items": len(state["dataset_items"]) if state["dataset_items"] else 0,
            "ingested_banks": len(state["ingested_banks"]),
        }

        # Test LLM connectivity if initialized
        if state["initialized"]:
            try:
                from hindsight_api.engine.llm_wrapper import create_llm_provider
                from hindsight_api.config import get_config
                config = get_config()
                llm = create_llm_provider(config)
                run_async(llm.verify_connection())
                info["llm_connected"] = True
                run_async(llm.cleanup())
            except Exception as e:
                info["llm_connected"] = False
                info["llm_error"] = str(e)

        return jsonify(info)

    # ------------------------------------------------------------------
    # POST /api/index  –  ingest LongMemEval sessions into memory banks
    # ------------------------------------------------------------------
    @app.route("/api/index", methods=["POST"])
    def index_sessions():
        """Ingest LongMemEval sessions into Hindsight memory banks.

        Request body (JSON):
            max_items:    int  – max dataset items to ingest (default: all)
            question_ids: list – specific question IDs to ingest (optional)
            force:        bool – re-ingest even if already done (default: false)
        """
        state = app.extensions["hindsight"]
        if not state["initialized"]:
            return jsonify({"error": "Engine not initialized. POST /api/init first."}), 400

        body = request.get_json(silent=True) or {}
        max_items = body.get("max_items")
        question_ids = body.get("question_ids")
        force = body.get("force", False)

        dataset = state["dataset"]
        items = state["dataset_items"]
        memory = state["memory"]

        if not items:
            return jsonify({"error": "Dataset not loaded"}), 500

        # Filter items
        if question_ids:
            items = [it for it in items if dataset.get_item_id(it) in set(question_ids)]
        if max_items:
            items = items[:max_items]

        from hindsight_api.models import RequestContext

        results = []
        total_sessions = 0
        skipped = 0
        start = time.time()

        for i, item in enumerate(items):
            item_id = dataset.get_item_id(item)
            bank_id = f"longmemeval_{item_id}"

            if bank_id in state["ingested_banks"] and not force:
                skipped += 1
                continue

            batch_contents = dataset.prepare_sessions_for_ingestion(item)
            num_sessions = len(batch_contents)

            try:
                # Clear existing data first
                run_async(memory.delete_bank(bank_id, request_context=RequestContext()))
                # Ingest
                run_async(memory.retain_batch_async(
                    bank_id=bank_id,
                    contents=batch_contents,
                    request_context=RequestContext(),
                ))
                state["ingested_banks"].add(bank_id)
                total_sessions += num_sessions
                results.append({
                    "item_id": item_id,
                    "bank_id": bank_id,
                    "sessions_ingested": num_sessions,
                    "status": "ok",
                })
                logger.info(f"[{i+1}/{len(items)}] Ingested {num_sessions} sessions for {item_id}")
            except Exception as e:
                logger.exception(f"Failed to ingest {item_id}")
                results.append({
                    "item_id": item_id,
                    "bank_id": bank_id,
                    "status": "error",
                    "error": str(e),
                })

        elapsed = time.time() - start
        return jsonify({
            "status": "completed",
            "items_processed": len(results),
            "items_skipped": skipped,
            "total_sessions_ingested": total_sessions,
            "elapsed_seconds": round(elapsed, 2),
            "details": results,
        })

    # ------------------------------------------------------------------
    # POST /api/retrieve  –  query a memory bank
    # ------------------------------------------------------------------
    @app.route("/api/retrieve", methods=["POST"])
    def retrieve():
        """Retrieve memories for a query.

        Request body (JSON):
            question:      str  – the query text (required)
            question_id:   str  – LongMemEval question ID (required, used as bank prefix)
            question_date: str  – ISO-8601 date for temporal context (optional)
            max_tokens:    int  – max tokens to retrieve (default: 8192)
            budget:        str  – search budget: low/mid/high (default: mid)
        """
        state = app.extensions["hindsight"]
        if not state["initialized"]:
            return jsonify({"error": "Engine not initialized. POST /api/init first."}), 400

        body = request.get_json(silent=True) or {}
        question = body.get("question")
        question_id = body.get("question_id")
        if not question or not question_id:
            return jsonify({"error": "Both 'question' and 'question_id' are required."}), 400

        max_tokens = body.get("max_tokens", 8192)
        budget_str = body.get("budget", "mid").lower()
        question_date_str = body.get("question_date")

        from hindsight_api.engine.memory_engine import Budget
        from hindsight_api.models import RequestContext

        budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
        budget = budget_map.get(budget_str, Budget.MID)

        question_date = None
        if question_date_str:
            try:
                question_date = datetime.fromisoformat(question_date_str)
                if question_date.tzinfo is None:
                    question_date = question_date.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        bank_id = f"longmemeval_{question_id}"
        memory = state["memory"]

        try:
            start = time.time()
            search_result = run_async(memory.recall_async(
                bank_id=bank_id,
                query=question,
                budget=budget,
                max_tokens=max_tokens,
                question_date=question_date,
                include_entities=True,
                max_entity_tokens=2048,
                include_chunks=True,
                request_context=RequestContext(),
            ))
            elapsed = time.time() - start

            recall_dict = search_result.model_dump()

            return jsonify({
                "status": "ok",
                "bank_id": bank_id,
                "question": question,
                "num_results": len(search_result.results) if search_result.results else 0,
                "num_entities": len(search_result.entities) if search_result.entities else 0,
                "num_chunks": len(search_result.chunks) if search_result.chunks else 0,
                "elapsed_seconds": round(elapsed, 2),
                "recall_result": recall_dict,
            })
        except Exception as e:
            logger.exception(f"Retrieval failed for {bank_id}")
            return jsonify({"status": "error", "error": str(e), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # POST /api/answer  –  retrieve + generate answer
    # ------------------------------------------------------------------
    @app.route("/api/answer", methods=["POST"])
    def answer():
        """Retrieve memories and generate an answer.

        Request body (JSON):
            question:      str  – the query (required)
            question_id:   str  – LongMemEval question ID (required)
            question_date: str  – ISO-8601 date (optional)
            max_tokens:    int  – max recall tokens (default: 8192)
            budget:        str  – low/mid/high (default: mid)
            context_format: str – json/structured (default: json)
        """
        state = app.extensions["hindsight"]
        if not state["initialized"]:
            return jsonify({"error": "Engine not initialized. POST /api/init first."}), 400

        body = request.get_json(silent=True) or {}
        question = body.get("question")
        question_id = body.get("question_id")
        if not question or not question_id:
            return jsonify({"error": "Both 'question' and 'question_id' are required."}), 400

        max_tokens = body.get("max_tokens", 8192)
        budget_str = body.get("budget", "mid").lower()
        context_format = body.get("context_format", "json")
        question_date_str = body.get("question_date")
        question_type = body.get("question_type")

        from hindsight_api.engine.memory_engine import Budget
        from hindsight_api.models import RequestContext
        from benchmarks.longmemeval.longmemeval_benchmark import LongMemEvalAnswerGenerator

        budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
        budget = budget_map.get(budget_str, Budget.MID)

        question_date = None
        if question_date_str:
            try:
                question_date = datetime.fromisoformat(question_date_str)
                if question_date.tzinfo is None:
                    question_date = question_date.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        bank_id = f"longmemeval_{question_id}"
        memory = state["memory"]

        try:
            start = time.time()

            # Recall
            search_result = run_async(memory.recall_async(
                bank_id=bank_id,
                query=question,
                budget=budget,
                max_tokens=max_tokens,
                question_date=question_date,
                include_entities=True,
                max_entity_tokens=2048,
                include_chunks=True,
                request_context=RequestContext(),
            ))
            recall_time = time.time() - start

            if not search_result.results:
                return jsonify({
                    "status": "ok",
                    "answer": "I don't have enough information to answer that question.",
                    "reasoning": "No relevant memories found.",
                    "recall_time_seconds": round(recall_time, 2),
                    "num_results": 0,
                })

            recall_dict = search_result.model_dump()

            # Generate answer
            generator = LongMemEvalAnswerGenerator(context_format=context_format)
            gen_start = time.time()
            answer_text, reasoning, memories_override = run_async(
                generator.generate_answer(question, recall_dict, question_date, question_type, bank_id=bank_id)
            )
            gen_time = time.time() - gen_start

            return jsonify({
                "status": "ok",
                "answer": answer_text,
                "reasoning": reasoning,
                "recall_time_seconds": round(recall_time, 2),
                "generation_time_seconds": round(gen_time, 2),
                "num_results": len(search_result.results),
                "num_entities": len(search_result.entities) if search_result.entities else 0,
            })
        except Exception as e:
            logger.exception(f"Answer generation failed for {bank_id}")
            return jsonify({"status": "error", "error": str(e), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # POST /api/benchmark/run  –  full E2E benchmark
    # ------------------------------------------------------------------
    @app.route("/api/benchmark/run", methods=["POST"])
    def run_benchmark_endpoint():
        """Run the full LongMemEval benchmark E2E.

        Request body (JSON):
            max_items:                int  – max items to evaluate (default: all)
            max_questions_per_item:   int  – max questions per item (default: all)
            thinking_budget:          int  – search budget (default: 500)
            max_tokens:               int  – max recall tokens (default: 8192)
            skip_ingestion:           bool – skip ingestion phase (default: false)
            context_format:           str  – json/structured (default: json)
            results_filename:         str  – output filename (default: benchmark_results.json)
            max_concurrent_items:     int  – parallel items (default: 1)
            category:                 str  – filter by question type (optional)
        """
        state = app.extensions["hindsight"]
        if not state["initialized"]:
            return jsonify({"error": "Engine not initialized. POST /api/init first."}), 400

        body = request.get_json(silent=True) or {}

        try:
            from benchmarks.longmemeval.longmemeval_benchmark import run_benchmark

            start = time.time()
            results = run_async(run_benchmark(
                max_instances=body.get("max_items"),
                max_questions_per_instance=body.get("max_questions_per_item"),
                thinking_budget=body.get("thinking_budget", 500),
                max_tokens=body.get("max_tokens", 8192),
                skip_ingestion=body.get("skip_ingestion", False),
                context_format=body.get("context_format", "json"),
                results_filename=body.get("results_filename", "benchmark_results.json"),
                max_concurrent_items=body.get("max_concurrent_items", 1),
                category=body.get("category"),
            ))
            elapsed = time.time() - start

            # Extract summary metrics
            summary = {}
            if results:
                summary = {
                    "total_items": results.get("total_items", 0),
                    "overall_metrics": results.get("overall_metrics", {}),
                    "elapsed_seconds": round(elapsed, 2),
                }

            return jsonify({
                "status": "completed",
                "summary": summary,
                "full_results": results,
            })
        except Exception as e:
            logger.exception("Benchmark run failed")
            return jsonify({"status": "error", "error": str(e), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # GET /api/dataset/info  –  dataset statistics
    # ------------------------------------------------------------------
    @app.route("/api/dataset/info", methods=["GET"])
    def dataset_info():
        """Return statistics about the loaded LongMemEval dataset."""
        state = app.extensions["hindsight"]
        items = state["dataset_items"]
        dataset = state["dataset"]

        if not items:
            return jsonify({"error": "Dataset not loaded. POST /api/init first."}), 400

        # Count by category
        from collections import Counter
        categories = Counter(it.get("question_type", "unknown") for it in items)

        # Sample item structure
        sample = items[0] if items else None
        sample_id = dataset.get_item_id(sample) if sample and dataset else None
        sample_sessions = len(sample.get("haystack_sessions", [])) if sample else 0

        return jsonify({
            "total_items": len(items),
            "categories": dict(categories),
            "sample_item_id": sample_id,
            "sample_num_sessions": sample_sessions,
            "ingested_banks": len(state["ingested_banks"]),
        })

    # ------------------------------------------------------------------
    # GET /api/dataset/item/<question_id>  –  single item details
    # ------------------------------------------------------------------
    @app.route("/api/dataset/item/<question_id>", methods=["GET"])
    def dataset_item(question_id: str):
        """Return details for a specific dataset item."""
        state = app.extensions["hindsight"]
        items = state["dataset_items"]
        dataset = state["dataset"]

        if not items:
            return jsonify({"error": "Dataset not loaded"}), 400

        item = next((it for it in items if dataset.get_item_id(it) == question_id), None)
        if not item:
            return jsonify({"error": f"Item '{question_id}' not found"}), 404

        qa_pairs = dataset.get_qa_pairs(item)
        sessions = item.get("haystack_sessions", [])

        return jsonify({
            "question_id": question_id,
            "question_type": item.get("question_type"),
            "num_sessions": len(sessions),
            "num_qa_pairs": len(qa_pairs),
            "qa_pairs": qa_pairs,
            "haystack_dates": item.get("haystack_dates", []),
            "bank_id": f"longmemeval_{question_id}",
            "is_ingested": f"longmemeval_{question_id}" in state["ingested_banks"],
        })

    return app


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        LongMemEval ↔ Hindsight Flask Bridge                 ║
╠══════════════════════════════════════════════════════════════╣
║  Provider : {os.getenv('HINDSIGHT_API_LLM_PROVIDER', 'not set'):<46s}  ║
║  Model    : {os.getenv('HINDSIGHT_API_LLM_MODEL', 'not set'):<46s}  ║
║  Azure EP : {os.getenv('HINDSIGHT_API_LLM_AZURE_ENDPOINT', 'not set'):<46s}  ║
║  Entra ID : {os.getenv('HINDSIGHT_API_LLM_AZURE_USE_ENTRA_ID', 'not set'):<46s}  ║
║  Port     : {str(port):<46s}  ║
╚══════════════════════════════════════════════════════════════╝

Endpoints:
  POST /api/init               – Initialize memory engine
  GET  /api/health             – Health check + LLM connectivity
  GET  /api/dataset/info       – Dataset statistics
  GET  /api/dataset/item/<id>  – Single item details
  POST /api/index              – Ingest sessions into memory
  POST /api/retrieve           – Query memories (recall)
  POST /api/answer             – Retrieve + generate answer
  POST /api/benchmark/run      – Full E2E benchmark
""")

    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=debug)