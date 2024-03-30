import json
import os
import sqlite3
import sys
import time

import chdb
import duckdb
import matplotlib.pyplot as plt
import monetdbe
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from chdb.session import Session

from util import benchmark

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "generator"))
from generator.parquet_generator import generate_file


def single_benchmark(
    monetdb_con,
    duckdb_con,
    clickhouse_con,
    sqlite_con,
    key_names,
    warmup=2,
    iterations=80,
):
    res = {}
    order_str = ",".join(key_names)
    query = f"SELECT * FROM test ORDER BY {order_str}"
    num_attr = len(key_names)
    res["MonetDB"] = benchmark(
        "MonetDB Sort",
        num_attr,
        monetdb_sort,
        warmup,
        iterations,
        monetdb_con,
        query,
    )
    res["DuckDB"] = benchmark(
        "DuckDB Sort",
        num_attr,
        duckdb_sort,
        warmup,
        iterations,
        duckdb_con,
        query,
    )
    res["ClickHouse"] = benchmark(
        "ClickHouse Sort",
        num_attr,
        clickhouse_sort,
        warmup,
        iterations,
        clickhouse_con,
        query,
    )
    res["SQLite"] = benchmark(
        "SQLite Sort",
        num_attr,
        sqlite_sort,
        warmup,
        iterations,
        sqlite_con,
        query,
    )
    return res


#                                           MonetDB Relating Functions
def monetdb_prepare(table_name, creation_query, column_names, parquet_file):
    # Read parquet
    df = pq.read_table("out.parquet").to_pandas(integer_object_nulls=True)

    # Create Table
    con = monetdbe.connect()
    try:
        con.execute(creation_query)
    except:
        con.execute(f"DROP TABLE {table_name}")
        con.execute(creation_query)

    # Insert values into table
    column_str = f"( {', '.join(column_names)} )"
    insertion_query = monetdb_prepare_insert_query(df, column_str)

    # Load data into the database
    con.execute(insertion_query)
    return con


def monetdb_prepare_insert_query(df, column_str):
    query = f"INSERT INTO test {column_str} VALUES "
    value_queries = []
    for i in range(len(df)):
        values = df.iloc[i].values
        # Create a list to hold the formatted values for the current row
        formatted_values = []
        for value in values:
            if value is None or pd.isna(value):
                formatted_values.append("NULL")
            elif isinstance(value, str):
                formatted_value = value.replace("'", "''")
                formatted_values.append(f"'{formatted_value}'")
            else:
                formatted_values.append(str(value))
        value_queries.append(f"({', '.join(formatted_values)})")
    query += ", ".join(value_queries)
    return query


def monetdb_sort(con, query):
    res = con.execute(query)
    return res


#                                            DuckDB Relating Functions
def duckdb_preapre(parquet_file, table_name):
    con = duckdb.connect()
    query = f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_file}')"
    con.execute(query)
    return con


def duckdb_sort(con, query):
    res = con.execute(query)
    return res


#                                            ClickHouse Relating Functions
def click_house_prepare(create_query, parquet_file, table_name):
    session = Session()
    session.query("CREATE DATABASE IF NOT EXISTS benchmark")
    session.query("USE benchmark")
    data_load_click_house(session, table_name, create_query, parquet_file)
    return session


def data_load_click_house(session, table_name, create_query, parquet_file):
    """
    TODO: Current generated parquet can not be loaded into clickhouse directly.
    For now, a temp CSV file is generated and loaded into the clickhouse.
    """
    df = pq.read_table(parquet_file).to_pandas(integer_object_nulls=True)
    df.to_csv("out.csv", index=False)
    rest_query = " AS SELECT * FROM file('out.csv')"
    create_query = create_query + rest_query
    try:
        session.query(create_query)
    except:
        session.query(f"DROP TABLE {table_name}")
        session.query(create_query)
    # Delete temp file
    os.remove("out.csv")


def clickhouse_sort(con, query):
    res = con.query(query)
    return res


#                                           SQLite Relating Functions
def sqlite_prepare(parquet_file, table_name):
    con = sqlite3.connect(":memory:")
    df = pq.read_table("out.parquet").to_pandas(integer_object_nulls=True)
    df.to_sql("test", con, if_exists="replace", index=False)
    return con


def sqlite_sort(con, query):
    res = con.execute(query)
    return res


def change_config_size(config_path, size):
    with open(config_path, "r") as f:
        json_data = json.load(f)
    if json_data is None:
        print(f"Failed to load JSON file at {config_path}.")
        exit(-1)
    json_data["size"] = size
    # Indent to be true
    with open(config_path, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    os.nice(-20)

    #               DB Preparations
    test_size = 50000
    config_name = "test_use_config.json"
    parquet_file = "out.parquet"
    change_config_size(config_name, test_size)
    generate_file(config_name, parquet_file)

    # ClickHouse
    click_house_creation_query = """
        CREATE TABLE test (
                key Int32
                , strDup VARCHAR(255)
                , intNumDup Int32
                , intNum Int32
                , doubleNum Double)
        ENGINE MergeTree
        ORDER BY key
        """
    click_house_session = click_house_prepare(
        click_house_creation_query, parquet_file, "test"
    )
    # DuckDB
    duckdb_con = duckdb_preapre(parquet_file, "test")
    # MonetDB
    monetdb_creation_query = """
        CREATE TABLE test (
                key Int
                , strDup VARCHAR(255)
                , intNumDup Int
                , intNum Int
                , doubleNum Double)
        """
    monetdb_con = monetdb_prepare(
        "test",
        monetdb_creation_query,
        ["key", "strDup", "intNumDup", "intNum", "doubleNum"],
        parquet_file,
    )
    # SQLite
    sqlite_con = sqlite_prepare(parquet_file, "test")

    #                                   Benchmarks
    # Single Keys
    res_single = single_benchmark(
        monetdb_con, duckdb_con, click_house_session, sqlite_con, ["intNum"]
    )
    # Double Keys
    res_double = single_benchmark(
        monetdb_con, duckdb_con, click_house_session, sqlite_con, ["strDup", "intNum"]
    )
    # Triple Keys
    res_triple = single_benchmark(
        monetdb_con,
        duckdb_con,
        click_house_session,
        sqlite_con,
        ["strDup", "intNumDup", "intNum"],
    )
    # Four Keys
    res_four = single_benchmark(
        monetdb_con,
        duckdb_con,
        click_house_session,
        sqlite_con,
        ["strDup", "intNumDup", "intNum", "doubleNum"],
    )
    results = [res_single, res_double, res_triple, res_four]
    num_attr = [1, 2, 3, 4]
    duckdb_res = [result["DuckDB"]["median"] for result in results]
    monetdb_res = [result["MonetDB"]["median"] for result in results]
    clickhouse_res = [result["ClickHouse"]["median"] for result in results]
    sqlite_res = [result["SQLite"]["median"] for result in results]
    # Plot
    plt.plot(num_attr, duckdb_res, label="DuckDB", color="blue", marker="o")
    plt.plot(num_attr, monetdb_res, label="MonetDB", color="red", marker="s")
    plt.plot(num_attr, clickhouse_res, label="ClickHouse", color="green", marker="^")
    plt.plot(num_attr, sqlite_res, label="SQLite", color="purple", marker="x")
    plt.xlabel("Number of Attributes")
    plt.ylabel("Average Query Time (ms)")
    plt.title("Average Query Time by Number of Attributes")
    plt.xticks(num_attr)
    plt.legend()
    plt.savefig("benchmark_key_comparison.png")

    # Create a temp file to store the results
    # Append <size,name,res>
    with open("benchmark_key_comparison.json", "w") as f:
        json.dump(
            {
                "size": test_size,
                "DuckDB": duckdb_res[3],
                "MonetDB": monetdb_res[3],
                "ClickHouse": clickhouse_res[3],
                "SQLite": sqlite_res[3],
            },
            f,
        )
