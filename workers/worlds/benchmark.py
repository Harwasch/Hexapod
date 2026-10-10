"""Benchmark an existing worker: explicitly starts inference, never provisions a GPU.

WORLD_GATEWAY_TOKEN=... python benchmark.py --url https://worker.example --seconds 60
Deletes its anonymous session on exit. Reports worker generation and client delivery
separately; optional --hourly-price is an operator estimate, never a measured bill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

from preflight import WorkerClient, WorkerError, inspect_health


def number(value):
    try:
        return (
            float(value)
            if isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            else None
        )
    except OverflowError:
        return None


def generation_metrics(session):
    """Use cumulative fields only; legacy generatedFPS is merely the latest chunk."""
    frames = number(session.get("totalGeneratedFrames"))
    seconds = number(session.get("totalGenerationSeconds"))
    return {
        "generatedFrames": frames,
        "generationSeconds": seconds,
        "generatedFPS": frames / seconds
        if frames is not None and seconds is not None and seconds > 0
        else None,
        "modelLoadSeconds": number(session.get("modelLoadSeconds")),
        "lastChunkGeneratedFPS": number(session.get("generatedFPS")),
        "scope": "worker_reported_sampling_and_encoding_excludes_initial_model_load",
    }


def run_benchmark(
    client,
    *,
    seconds=60,
    model_id="astronex-world",
    seed=42,
    hourly_price=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    if not 1 <= seconds <= 3600:
        raise WorkerError("invalid_benchmark_duration")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", model_id):
        raise WorkerError("invalid_model_id")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise WorkerError("invalid_seed")
    if hourly_price is not None and (
        number(hourly_price) is None or hourly_price > 1000
    ):
        raise WorkerError("invalid_hourly_price")
    result = {
        "schemaVersion": 2,
        "kind": "worlds-worker-benchmark",
        "passed": False,
        "measured": True,
        "modelId": model_id,
        "seed": seed,
        "computeProvisioned": False,
        "samplingHz": 10,
        "cleanup": "not_needed",
    }
    unique = set()
    unidentified = set()
    first_frame = None
    final = {}
    created = False
    identifier = str(uuid.uuid4())
    started = clock()
    measurement_finished = started
    try:
        health = inspect_health(
            client.request("GET", "/health")[2], model_id, getattr(client, "token", "")
        )
        if not health["ready"]:
            raise WorkerError("selected_model_unavailable")
        result["model"] = health
        started = clock()
        # Cleanup uses our anonymous ID even if creation reached the worker but its
        # response timed out, leaving the client uncertain whether inference started.
        created = True
        client.request(
            "POST",
            "/sessions",
            {
                "id": identifier,
                "modelId": model_id,
                "input": {
                    "prompt": "First-person view of a quiet woodland path in daylight."
                },
                "seed": seed,
                "quality": "balanced",
            },
        )
        next_heartbeat = clock()
        while clock() - started < seconds:
            if clock() >= next_heartbeat:
                client.request("POST", f"/sessions/{identifier}/heartbeat", {"active": True})
                next_heartbeat = clock() + 20
            status, headers, frame = client.request(
                "GET", f"/sessions/{identifier}/frame"
            )
            if status == 200:
                content_type = headers.get("Content-Type", "").split(";")[0].lower()
                if (
                    not isinstance(frame, bytes)
                    or not frame
                    or not (
                        content_type == "image/jpeg"
                        and frame.startswith(b"\xff\xd8\xff")
                        or content_type == "image/png"
                        and frame.startswith(b"\x89PNG\r\n\x1a\n")
                        or content_type == "image/webp"
                        and frame.startswith(b"RIFF")
                        and frame[8:12] == b"WEBP"
                    )
                ):
                    raise WorkerError("invalid_delivered_frame")
                index = headers.get("X-Frame-Index")
                if index is not None and str(index).isdigit():
                    unique.add(str(index))
                else:
                    unidentified.add(hashlib.sha256(frame).hexdigest())
                if first_frame is None:
                    first_frame = clock() - started
            elif status != 204:
                raise WorkerError("invalid_frame_response")
            final = client.request("GET", f"/sessions/{identifier}")[2]
            if not isinstance(final, dict):
                final = {}
                raise WorkerError("invalid_session_status")
            if final.get("status") in ("error", "failed"):
                raise WorkerError("model_inference_failed")
            sleep(min(0.1, max(0, seconds - (clock() - started))))
        if not unique and not unidentified:
            raise WorkerError("no_frames_delivered")
        result["passed"] = True
    except WorkerError as error:
        result["failure"] = error.code
    finally:
        measurement_finished = clock()
        if created:
            try:
                client.request("DELETE", f"/sessions/{identifier}")
                result["cleanup"] = "session_deleted"
            except WorkerError as error:
                if error.code == "worker_http_404":
                    result["cleanup"] = "session_absent"
                else:
                    result["cleanup"] = "failed_stop_session_on_worker"
                    result["passed"] = False
                    result.setdefault("failure", "session_cleanup_failed")
    duration = max(0, measurement_finished - started)
    result["delivery"] = {
        "durationSeconds": duration,
        "timeToFirstFrameSeconds": first_frame,
        "uniqueIndexedFrames": len(unique),
        "distinctUnindexedImages": len(unidentified),
        "observedIndexedDeliveryFPS": len(unique) / duration if duration else 0,
        "note": "Client sampling is capped at 10 Hz. Delivery FPS is not model generation FPS; unindexed images are a lower bound.",
    }
    result["generation"] = generation_metrics(final)
    result["measured"] = bool(
        unique or unidentified or result["generation"]["generatedFrames"]
    )
    result["pricing"] = (
        {
            "kind": "estimate",
            "source": "operator_supplied_hourly_price",
            "hourlyPriceUSD": hourly_price,
            "estimatedSessionCostUSD": hourly_price * max(0, clock() - started) / 3600
            if created
            else 0,
            "note": "Wall-time estimate including cleanup, excluding worker idle time, storage and network fees; not a measured charge.",
        }
        if hourly_price is not None
        else {"kind": "not_measured"}
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("benchmark-result.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="astronex-world")
    parser.add_argument("--hourly-price", type=float)
    args = parser.parse_args(argv)
    try:
        client = WorkerClient(
            args.url, os.environ.get("WORLD_GATEWAY_TOKEN", ""), timeout=30
        )
        result = run_benchmark(
            client,
            seconds=args.seconds,
            model_id=args.model,
            seed=args.seed,
            hourly_price=args.hourly_price,
        )
    except WorkerError as error:
        result = {
            "schemaVersion": 2,
            "kind": "worlds-worker-benchmark",
            "passed": False,
            "measured": False,
            "failure": error.code,
        }
    try:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except OSError:
        result["passed"] = False
        result["failure"] = "report_output_unwritable"
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
