#!/usr/bin/env python3
import json
import time
from argparse import ArgumentParser

import pyarrow as pa
import pyarrow.parquet as pq

from column_meta import ColumnMeta
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


def generate_logical_data(metas, size):
    data = {}
    for meta in metas:
        data[meta.name] = generate_column_data(meta, size)
    return data


def generate_physical_data(metas, logical_data, output_path):
    schemas = [meta.to_pyarrow_schema() for meta in metas]
    parquet_schema = pa.schema(schemas)
    # Configure compression and encodings for each columns
    compressions = {
        meta.name: meta.compression for meta in metas if meta.compression is not None
    }
    encodings, dictionary, byte_stream = get_encodings(metas)
    table = pa.table(logical_data, schema=parquet_schema)
    # Write the table to a Parquet file
    with pa.OSFile(output_path, "wb") as f:
        writer = pq.ParquetWriter(
            f,
            parquet_schema,
            compression=compressions,
            use_dictionary=dictionary,
            column_encoding=encodings,
            use_byte_stream_split=byte_stream,
        )
        writer.write_table(table)
        writer.close()


def get_encodings(metas):
    """
    Generate required encoding parameters for parquet writer based on column metadata.
    """
    dictionaries = {}
    encodings = {}
    byte_stream = {}
    # Iterate over metas and populate the dictionaries and encodings
    for meta in metas:
        if meta.encoding == "PLAIN_DICTIONARY":
            dictionaries[meta.name] = True
        elif meta.encoding == "BYTE_STREAM_SPLIT":
            byte_stream[meta.name] = True
        else:
            encodings[meta.name] = meta.encoding
    return encodings, dictionaries, byte_stream


def generate_file(file_name, out_name):
    with open(file_name, "r") as f:
        json_data = json.load(f)
    if json_data is None:
        print(f"Failed to load JSON file at {file_name}.")
        exit(-1)

    # Generate the Parquet file
    column_metas = []
    for col in json_data["columns"]:
        column_metas.append(ColumnMeta.from_json(col))
    size = json_data["size"]
    logical_data = generate_logical_data(column_metas, size)
    generate_physical_data(column_metas, logical_data, out_name)


if __name__ == "__main__":
    args = parse_args()
    if args.json is None or args.output is None:
        print("Usage: python parquet_generator.py -j <config_path> -o <output_path>")
        exit(-1)
    print(f"Generating Parquet file at {args.output} using {args.json}.")

    generate_file(args.json, args.output)
