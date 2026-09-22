"""Controlled offline service experiments; these are NOT model/GPU benchmarks."""
import argparse
import asyncio
import importlib.metadata
import json
import platform
import socket
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving_lab.app import create_app  # noqa: E402
from serving_lab.config import Settings  # noqa: E402


async def run(port=8788):
    settings = Settings(enable_faults=True, concurrency=2, max_queue=2, queue_timeout=0.03,
                        request_timeout=0.8, offline_delay=0.02, breaker_threshold=2,
                        breaker_reset=0.1, rate_burst=1000)
    app = create_app(settings)
    sock = socket.socket()
    sock.bind(("127.0.0.1", port))
    sock.listen()
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on"))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if serving.done():
            await serving
        await asyncio.sleep(0.01)
    base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    try:
        async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=2) as client:
            async def request(name, *, fault=None, stream=False):
                body = {"model": "offline-small", "max_tokens": 4, "stream": stream,
                        "messages": [{"role": "user", "content": "Fictional public benchmark example"}]}
                if fault:
                    body["fault"] = fault
                started, first, chunks = time.perf_counter(), None, 0
                async with client.stream("POST", "/v1/chat/completions", json=body) as response:
                    if stream and response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.startswith("data: {") and '"content"' in line:
                                first = first or time.perf_counter()
                                chunks += 1
                    else:
                        await response.aread()
                    return {"scenario": name, "http_status": response.status_code,
                            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                            "ttft_ms": round((first - started) * 1000, 3) if first else None,
                            "content_chunks": chunks}
            records = [await request("baseline", stream=True) for _ in range(5)]
            records.extend(await asyncio.gather(*(request("backpressure", fault={"delay_ms": 25}) for _ in range(12))))
            records.append(await request("timeout", fault={"delay_ms": 500}))
            records.append(await request("injected_failure", fault={"fail_after": 0}))
            records.append(await request("circuit_open"))
            await asyncio.sleep(0.12)
            records.append(await request("recovery", stream=True))
            runtime = app.state.runtime
            capacity = runtime.capacities["offline-small"]
            summary = {}
            for name in sorted({item["scenario"] for item in records}):
                selected = [item for item in records if item["scenario"] == name]
                durations = sorted(item["duration_ms"] for item in selected)
                summary[name] = {"requests": len(selected), "statuses": {str(code): sum(item["http_status"] == code for item in selected) for code in sorted({item["http_status"] for item in selected})},
                                 "latency_p50_ms": round(statistics.median(durations), 3),
                                 "latency_max_ms": max(durations)}
            report = {"experiment": "offline deterministic service reliability; no GPU/model throughput claims",
                      "recorded_at_utc": datetime.now(UTC).isoformat(), "python": platform.python_version(),
                      "platform": platform.platform(),
                      "packages": {name: importlib.metadata.version(name) for name in ("fastapi", "httpx", "uvicorn")},
                      "settings": {"concurrency": 2, "max_queue": 2, "queue_timeout_ms": 30, "request_timeout_ms": 800},
                      "summary": summary, "records": records,
                      "resource_invariants": {"peak_active": capacity.peak_active, "peak_queued": capacity.peak_waiting,
                                              "final_active": capacity.active, "final_queued": capacity.waiting,
                                              "final_open_streams": runtime.backends["offline-small"].open_streams}}
            assert report["resource_invariants"] == {"peak_active": 2, "peak_queued": 2, "final_active": 0, "final_queued": 0, "final_open_streams": 0}
            assert summary["timeout"]["statuses"] == {"504": 1}
            assert summary["injected_failure"]["statuses"] == {"502": 1}
            assert summary["circuit_open"]["statuses"] == {"503": 1}
            assert summary["recovery"]["statuses"] == {"200": 1}
            return report
    finally:
        server.should_exit = True
        await serving
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--output", type=Path, default=Path("results/offline-benchmark.json"))
    args = parser.parse_args()
    result = asyncio.run(run(args.port))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": result["summary"], "resources": result["resource_invariants"]}, indent=2))
