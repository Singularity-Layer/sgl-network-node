#!/usr/bin/env python3
"""Run one real xMESH `/v1/systemone` canary against a native llama-server."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_ID = "singularity-layer/Singularity-xMESH-0.8B"
MODEL_SHA256 = "9f4c08f19c4d203059bc74ed2754e800ec472d7276d8ddb2321480e86d392fff"

REQUEST = {
    "model": MODEL_ID,
    "state": "Summarize a supplied earnings-call transcript, add current company news, and store the final brief. Constraints: total_budget_usd=0.150000; preferred_network=mixed; verified_only=yes; human_approval=not-allowed.",
    "questions": {
        "need_acquire": {"type": "noul", "instructions": "Need a service to obtain raw source material? False if the caller supplies it or its URL."},
        "need_enrich": {"type": "noul", "instructions": "Does the request need outside web, market, chain, or reference data?"},
        "pick_enrich": {"type": "choice", "instructions": "Choose by capability, then I/O, constraints, and price.", "criteria": {"company_news_1": "cap=company news data; in=json; out=json; price=0.010000", "weather_1": "cap=weather forecast; in=json; out=json; price=0.005000"}},
        "need_analyze": {"type": "noul", "instructions": "Does the request need analysis, judgment, summary, rating, comparison, or written output?"},
        "pick_analyze": {"type": "choice", "instructions": "Choose by capability, then I/O, constraints, and price.", "criteria": {"summarizer_1": "cap=text summarization; in=text; out=json; price=0.030000", "translator_1": "cap=language translation; in=text; out=json; price=0.020000"}},
        "need_deliver": {"type": "noul", "instructions": "Must the result be sent, published, or stored outside this system?"},
        "pick_deliver": {"type": "choice", "instructions": "Choose by capability, then I/O, constraints, and price.", "criteria": {"storage_1": "cap=object storage; in=text; out=json; price=0.010000", "email_1": "cap=email delivery; in=json; out=json; price=0.001000"}},
        "needs_clarification": {"type": "noul", "instructions": "Is an essential capability or outcome missing?"},
        "has_valid_flow": {"type": "noul", "instructions": "Does every required stage have a matching candidate?"},
    },
}

EXPECTED = {
    "need_acquire": False,
    "need_enrich": True,
    "pick_enrich": "company_news_1",
    "need_analyze": True,
    "pick_analyze": "summarizer_1",
    "need_deliver": False,
    "pick_deliver": "storage_1",
    "needs_clarification": False,
    "has_valid_flow": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, body: dict, timeout: float = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def normalize(answers: dict) -> dict:
    values = {}
    for name, answer in answers.items():
        if "choice" in answer:
            values[name] = answer["choice"]
        elif "score" in answer:
            values[name] = answer["score"]
        else:
            values[name] = float(answer["noul"]) >= 0.5
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()

    if sha256(args.model) != MODEL_SHA256:
        raise SystemExit("model SHA-256 mismatch")
    log_path = Path("xmesh-canary-llama.log")
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                str(args.server), "-m", str(args.model), "--host", "127.0.0.1",
                "--port", str(args.port), "-c", "2048", "-b", "2048", "-ub", "2048",
                "-np", "1", "-ngl", "0", "-t", "8", "-lv", "2",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            base = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited {process.returncode}: {log_path.read_text(errors='replace')[-2000:]}")
                try:
                    if get_json(f"{base}/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError, json.JSONDecodeError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("llama-server did not become ready")

            started = time.perf_counter()
            result = post_json(f"{base}/v1/systemone", REQUEST)
            elapsed_ms = (time.perf_counter() - started) * 1000
            actual = normalize(result["answers"])
            if actual != EXPECTED:
                raise RuntimeError(f"decision mismatch: {actual}")
            evidence = {
                "model_sha256": MODEL_SHA256,
                "input_tokens": result.get("usage", {}).get("input_tokens"),
                "matching_decisions": len(EXPECTED),
                "latency_ms": elapsed_ms,
                "answers": actual,
            }
            Path("xmesh-canary-result.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(json.dumps(evidence, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
