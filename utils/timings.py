from functools import wraps
from timeit import default_timer as timer


# Timing for iterate_gauss_seidel
def wtime(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        try:
            return fn(*args, **kwargs)
        finally:
            measure_time.timings.append(timer() - t1)

    measure_time.timings = []
    return measure_time
