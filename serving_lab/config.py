import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    mode: str = "offline"
    api_key: str = ""
    upstream_url: str = "http://127.0.0.1:8001"
    upstream_key: str = ""
    upstream_model: str = "Qwen/Qwen3-0.6B"
    concurrency: int = 2
    max_queue: int = 4
    queue_timeout: float = 0.25
    request_timeout: float = 5.0
    rate_per_second: float = 20.0
    rate_burst: int = 40
    breaker_threshold: int = 3
    breaker_reset: float = 2.0
    offline_delay: float = 0.01
    enable_faults: bool = False

    def __post_init__(self):
        if self.mode not in {"offline", "vllm"}:
            raise ValueError("MODE must be offline or vllm")
        if self.concurrency < 1 or self.max_queue < 0 or self.rate_burst < 1 or self.breaker_threshold < 1:
            raise ValueError("Invalid capacity configuration")
        if min(self.queue_timeout, self.request_timeout, self.rate_per_second, self.breaker_reset) <= 0:
            raise ValueError("Timeouts and rates must be positive")
        if self.offline_delay < 0:
            raise ValueError("OFFLINE_DELAY cannot be negative")
        parsed = urlparse(self.upstream_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("UPSTREAM_URL must be an HTTP(S) origin without credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("UPSTREAM_URL must be an origin, without path, query or fragment")
        if self.mode == "vllm" and not self.api_key:
            raise ValueError("API_KEY is required in vllm mode")
        if self.mode != "offline" and self.enable_faults:
            raise ValueError("Fault injection is restricted to offline mode")

    @classmethod
    def from_env(cls):
        # Deliberately no dotenv loading and no discovery of user credentials.
        defaults = cls()
        values = {}
        for name in defaults.__dataclass_fields__:
            raw = os.getenv(name.upper())
            if raw is None:
                continue
            default = getattr(defaults, name)
            if isinstance(default, bool):
                if raw.lower() not in {"true", "false", "1", "0"}:
                    raise ValueError(f"Invalid boolean: {name.upper()}")
                values[name] = raw.lower() in {"true", "1"}
            else:
                values[name] = type(default)(raw)
        return cls(**values)
