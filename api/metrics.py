from prometheus_client import Counter, Histogram
import time
from functools import wraps

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests",
    ["endpoint"])

# Decorator
def track_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        status = "success"

        try:
            result = func(*args, **kwargs)
            return result
        except Exception:
            status = "error"
            raise
        finally:
            latency = time.time() - start_time
            endpoint = func.__name__

            REQUEST_COUNT.labels(
                endpoint=endpoint,
                method="POST",
                status=status
            ).inc()

            REQUEST_LATENCY.labels(
                endpoint=endpoint
            ).observe(latency)

    return wrapper
