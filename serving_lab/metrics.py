import json
import logging
from collections import Counter, defaultdict

logger = logging.getLogger("serving_lab.requests")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


class Metrics:
    """Small Prometheus text exporter with a fixed, bounded label vocabulary."""

    buckets = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)

    def __init__(self):
        self.requests = Counter()
        self.chunks = Counter()
        self.latencies = defaultdict(lambda: [0] * len(self.buckets))
        self.duration_sum = Counter()
        self.duration_count = Counter()

    def observe(self, model, outcome, duration, request_id):
        self.requests[model, outcome] += 1
        self.duration_sum[model] += duration
        self.duration_count[model] += 1
        for index, bound in enumerate(self.buckets):
            self.latencies[model][index] += int(duration <= bound)
        # Never log body, generated content, raw URL, client identity or headers.
        logger.info(json.dumps({"request_id": request_id, "model": model, "outcome": outcome,
                                "duration_ms": round(duration * 1000, 3)}, separators=(",", ":")))

    def render(self, capacities, breakers):
        lines = ["# TYPE serving_requests_total counter"]
        for (model, outcome), count in sorted(self.requests.items()):
            lines.append(f'serving_requests_total{{model="{model}",outcome="{outcome}"}} {count}')
        lines.append("# TYPE serving_stream_chunks_total counter")
        for model, count in sorted(self.chunks.items()):
            lines.append(f'serving_stream_chunks_total{{model="{model}"}} {count}')
        lines.extend(["# TYPE serving_active_requests gauge", "# TYPE serving_queued_requests gauge",
                      "# TYPE serving_circuit_open gauge"])
        for model in sorted(capacities):
            lines.append(f'serving_active_requests{{model="{model}"}} {capacities[model].active}')
            lines.append(f'serving_queued_requests{{model="{model}"}} {capacities[model].waiting}')
            lines.append(f'serving_circuit_open{{model="{model}"}} {int(breakers[model].state != "closed")}')
        lines.append("# TYPE serving_request_duration_seconds histogram")
        for model in sorted(self.duration_count):
            for bound, count in zip(self.buckets, self.latencies[model], strict=True):
                lines.append(f'serving_request_duration_seconds_bucket{{model="{model}",le="{bound}"}} {count}')
            lines.append(f'serving_request_duration_seconds_bucket{{model="{model}",le="+Inf"}} {self.duration_count[model]}')
            lines.append(f'serving_request_duration_seconds_count{{model="{model}"}} {self.duration_count[model]}')
            lines.append(f'serving_request_duration_seconds_sum{{model="{model}"}} {self.duration_sum[model]}')
        return "\n".join(lines) + "\n"
