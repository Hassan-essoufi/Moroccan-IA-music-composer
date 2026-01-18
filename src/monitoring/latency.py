import time

def measure_latency(func):
    """
    Decorator to measure execution latency of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        latency = end_time - start_time
        print(f"[Latency] {func.__name__} executed in {latency:.4f} seconds")

        return result

    return wrapper


def measure_block_latency(start_time):
    """
    Measure latency in seconds for a code block.
    """
    return time.time() - start_time
