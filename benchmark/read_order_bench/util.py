import time

import numpy as np


def benchmark(discription, attr_num, benchmark_func, warmup, iterations, *args):
    # Warmup phase
    for i in range(warmup):
        benchmark_func(*args)

    times = []
    for i in range(iterations):
        start = time.perf_counter() * 1000
        res = benchmark_func(*args)
        end = time.perf_counter() * 1000
        times.append(end - start)

    times = np.array(times)
    min_time = np.min(times)
    max_time = np.max(times)
    avg_time = np.mean(times)
    median_time = np.median(times)
    std_time = np.std(times)

    # Calculate the percentiles for the average
    avg_percentile = _calculate_percentile(times, avg_time)

    print(
        f"Benchmark {discription} finished. Avg time: {avg_time} ms. Percentile: {avg_percentile}"
    )
    # Return a dictionary
    return {
        "Description": discription,
        "Number of Attributes": attr_num,
        "min": min_time,
        "max": max_time,
        "avg": avg_time,
        "median": median_time,
        "std": std_time,
        "avg_percentile": avg_percentile,
    }


# Calculate the percentile of the value in the array
def _calculate_percentile(arr, value):
    arr.sort()
    index = np.searchsorted(arr, value)
    percentile = index / len(arr)
    return percentile
