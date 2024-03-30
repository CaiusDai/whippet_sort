import os
import sys

import pyarrow.parquet as pq
from matplotlib import pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "generator"))
from generator.column_meta import ColumnMeta
from generator.parquet_generator import *

from util import benchmark


def arrow_read(file_name):
    parquet_table = pq.read_table(file_name)
    return parquet_table


def single_column_bench(scale, encoding, dtype, cardinality=None):
    params = {}
    params["min"] = 0
    params["max"] = 5 * scale
    str_length = None
    cardinality = scale // 50
    if dtype == "str":
        str_length = [20, 20]
    meta = ColumnMeta(
        "test",
        dtype,
        encoding=encoding,
        cardinality=cardinality,
        dist_params=params,
        str_length=str_length,
    )
    data = generate_logical_data([meta], scale)
    generate_physical_data([meta], data, "test.parquet")

    res = benchmark("Arrow Read", 1, arrow_read, 10, 50, "test.parquet")
    return res["median"]


if __name__ == "__main__":
    os.nice(-20)
    data_scale = [i for i in range(1000, 50000, 5000)]
    encodings = {}
    # Integer Bench
    encodings["int"] = ["PLAIN", "DELTA_BINARY_PACKED", "PLAIN_DICTIONARY"]
    # Str Bench
    encodings["str"] = [
        "PLAIN",
        "DELTA_BYTE_ARRAY",
        "DELTA_LENGTH_BYTE_ARRAY",
        "PLAIN_DICTIONARY",
    ]
    int_res = {}
    # Integer Bench
    for scale in data_scale:
        int_res[scale] = {}
        for encoding in encodings["int"]:
            int_res[scale][encoding] = single_column_bench(scale, encoding, "int")
    # String Bench
    str_res = {}
    for scale in data_scale:
        str_res[scale] = {}
        for encoding in encodings["str"]:
            str_res[scale][encoding] = single_column_bench(scale, encoding, "str")

    # Plotting code for int_read_bench.png
    plt.figure(figsize=(10, 6))
    for encoding in encodings["int"]:
        plt.plot(
            data_scale,
            [int_res[scale][encoding] for scale in data_scale],
            label=encoding,
        )
    plt.xlabel("Data Scale")
    plt.ylabel("Running Time (avg)")
    plt.title("Integer Read Benchmark")
    plt.legend()
    plt.savefig("int_read_bench.png")
    plt.close()

    # Plotting code for str_read_bench.png
    plt.figure(figsize=(10, 6))
    for encoding in encodings["str"]:
        plt.plot(
            data_scale,
            [str_res[scale][encoding] for scale in data_scale],
            label=encoding,
        )
    plt.xlabel("Data Scale")
    plt.ylabel("Running Time (avg)")
    plt.title("String Read Benchmark")
    plt.legend()
    plt.savefig("str_read_bench.png")
    plt.close()
