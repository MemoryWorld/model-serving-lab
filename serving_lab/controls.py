import asyncio
import time
from collections import OrderedDict


class Overloaded(Exception):
    pass


class CircuitOpen(Exception):
    pass


class Capacity:
    """Event-loop-local capacity, bounded waiting, cancellation-safe release."""

    def __init__(self, limit: int, max_queue: int, wait_seconds: float):
        self.limit, self.max_queue, self.wait_seconds = limit, max_queue, wait_seconds
        self.active = self.waiting = 0
        self.peak_active = self.peak_waiting = 0
        self._semaphore = asyncio.BoundedSemaphore(limit)

    async def acquire(self):
        queued = self.active >= self.limit or self.waiting > 0
        if queued and self.waiting >= self.max_queue:
            raise Overloaded("Queue is full")
        if queued:
            self.waiting += 1
            self.peak_waiting = max(self.peak_waiting, self.waiting)
        try:
            async with asyncio.timeout(self.wait_seconds):
                await self._semaphore.acquire()
        except TimeoutError as exc:
            raise Overloaded("Queue wait timed out") from exc
        finally:
            if queued:
                self.waiting -= 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        return Lease(self)


class Lease:
    def __init__(self, capacity: Capacity):
        self.capacity, self.closed = capacity, False

    def close(self):
        # Synchronous: task cancellation cannot interrupt local accounting.
        if not self.closed:
            self.closed = True
            self.capacity.active -= 1
            self.capacity._semaphore.release()


class CircuitBreaker:
    def __init__(self, threshold: int, reset_seconds: float, clock=time.monotonic):
        self.threshold, self.reset_seconds, self.clock = threshold, reset_seconds, clock
        self.failures, self.opened_at, self.probing = 0, None, False

    @property
    def state(self):
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.probing else "open"

    def acquire(self):
        if self.opened_at is not None:
            if self.probing or self.clock() - self.opened_at < self.reset_seconds:
                raise CircuitOpen()
            self.probing = True
        return CircuitTicket(self, self.probing)


class CircuitTicket:
    def __init__(self, breaker, probe):
        self.breaker, self.probe, self.done = breaker, probe, False

    def finish(self, outcome):
        if self.done:
            return
        self.done = True
        breaker = self.breaker
        if self.probe:
            breaker.probing = False
        if outcome == "success":
            # An older in-flight success must not close a newer open circuit.
            if self.probe or breaker.opened_at is None:
                breaker.failures, breaker.opened_at = 0, None
        elif outcome == "failure":
            breaker.failures += 1
            if self.probe or breaker.failures >= breaker.threshold:
                breaker.opened_at = breaker.clock()
        # Cancellation/queue rejection releases a half-open probe without
        # treating the client as a failed model invocation.


class TokenBucket:
    """Bounded per-client buckets; single-process lab limit, not a global quota."""

    def __init__(self, rate: float, burst: int, clock=time.monotonic, max_clients=1024):
        self.rate, self.burst, self.clock, self.max_clients = rate, burst, clock, max_clients
        self.clients = OrderedDict()

    def allow(self, identity: str):
        now = self.clock()
        tokens, last = self.clients.pop(identity, (float(self.burst), now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        allowed = tokens >= 1
        self.clients[identity] = (tokens - int(allowed), now)
        while len(self.clients) > self.max_clients:
            self.clients.popitem(last=False)
        return allowed
