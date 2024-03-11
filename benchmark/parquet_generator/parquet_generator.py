#!/usr/bin/env python3
from column_meta import ColumnMeta
import pyarrow as pa
import json
import time
from argparse import ArgumentParser
from data_generator import generate_column_data

workdir = "/workspace/whippet_docker"


def parse_args():
    parser = ArgumentParser(
        description="Generate Parquet files with different column metadata"
    )
    # Two inputs, json file and output path
    parser.add_argument("-j", "--json", type=str, required=True)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"{workdir}/data/parquet/{time.strftime('%Y%m%d%H%M%S')}.parquet",
    )
    args = parser.parse_args()
    return args


def generate_logical_data(metas):
    data = {}
    for meta in metas:
        data[meta.name] = generate_column_data(meta)
    return data


def generate_parquet_file(metas, logical_data, output_path):
    schemas = [meta.to_pyarrow_schema() for meta in metas]
    parquet_schema = pa.schema(schemas)
    # Configure compression and encodings for each columns
    compressions = [meta.compression for meta in metas]
    encodings = [meta.encoding for meta in metas]
    table = pa.table(logical_data, schema=parquet_schema)
    # Write the table to a Parquet file
    with pa.OSFile(output_path, "wb") as f:
        writer = pa.parquet.ParquetWriter(
            f, parquet_schema, compression=compressions, use_dictionary=encodings
        )
        writer.write_table(table)
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    if args.json is None:
        print("Usage: python parquet_generator.py -j <config_path> -o <output_path>")
        exit(-1)
    print(f"Generating Parquet file at {args.output} using {args.json}.")

    with open(args.json, "r") as f:
        json_data = json.load(f)
    if json_data is None:
        print(f"Failed to load JSON file at {args.json}.")
        exit(-1)

    # Generate the Parquet file
    column_metas = []
    for col in json_data["columns"]:
        column_metas.append(ColumnMeta.from_json(col))
    size = json_data["size"]
    logical_data = generate_logical_data(column_metas, size)
    exit(0)
