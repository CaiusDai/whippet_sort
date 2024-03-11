#!/usr/bin/env python3
import pyarrow as pa
import numpy as np


import pyarrow as pa
import numpy as np


class ColumnMeta:
    def __init__(
        self,
        name,
        dtype,
        distribution="normal",
        str_length=None,
        custom_generator=None,
        num_null=0,
        cardinality=None,
        num_repeated=1,
        dist_params={},
        fields=None,
        compression=None,
        encoding=None,
    ):
        """
        Initialize the ColumnMeta object with column metadata.

        Parameters:
        - name (str): The name of the column.
        - dtype (type or str): The data type of the column (e.g., int, str) or 'struct' 'list' for nested structures.
        - distribution (str): The name of the distribution to use for generating data. Not applicable for nested structures.
        - str_length (int): The length of the string. Only applicable for dtype 'str'.
        - custom_generator (callable): A custom function to generate column data. Takes precedence over distribution.
        - num_null(int): The number of null values in the column.
        - cardinality (int): The number of unique values in the column.
        - num_repeated(int): The number of times to repeat the column data.
        - dist_params (dict): Parameters for the distribution. Not applicable for 'struct' and 'list'.
        - fields (list of ColumnMeta): Child columns for a 'struct' or 'list' dtype.
        - compression (str): The compression algorithm to use for the column.
        - encoding (str): The encoding algorithm to use for the column.
        """
        # Input checks
        if dtype not in ["int", "double", "str", "struct"]:
            raise ValueError(f"Unsupported dtype: {dtype}.")
        if dtype == "struct" and fields is None:
            raise ValueError("Children must be provided for struct dtype.")
        if distribution not in ["normal", "uniform", "unique", "gamma", "random"]:
            raise ValueError(f"Invalid distribution configuration: {distribution}.")

        self.name = name
        self.dtype = dtype
        self.distribution = distribution
        self.str_length = str_length
        self.num_null = num_null
        self.cardinality = cardinality
        self.num_repeated = num_repeated
        self.custom_generator = custom_generator
        self.dist_params = dist_params
        self.fields = fields
        self.compression = compression
        self.encoding = encoding

    def to_pyarrow_schema(self):
        """
        Convert the ColumnMeta object to a PyArrow schema.
        """
        field_type = self.__map_to_pyarrow_type(self.dtype)
        nullable = self.num_null > 0
        field = pa.field(
            self.name,
            field_type,
            nullable=nullable,
        )
        field.set_compression(self.compression)
        return field

    def set_custom_generator(self, custom_generator):
        """
        Set the custom generator for the column.
        """
        self.custom_generator = custom_generator

    def get_compression_dic(self):
        """
        Get the compression dictionary for the column.
        """
        if self.dtype == "struct":
            return {field.name: field.get_compression_dic() for field in self.fields}
        else:
            return {self.name: self.compression}

    def get_encoding_dic(self):
        """
        Get the encoding dictionary for the column.
        """
        if self.dtype == "struct":
            return {field.name: field.get_encoding_dic() for field in self.fields}
        else:
            return {self.name: self.encoding}

    def __map_to_pyarrow_type(self):
        """
        Map the string representation of the dtype to the corresponding PyArrow type.
        """
        if self.num_repeated > 1:
            return pa.list_(pa.field(self.name, self.__map_to_pyarrow_type(self.dtype)))
        if self.dtype == "int":
            return pa.int64()
        elif self.dtype == "double":
            return pa.float64()
        elif self.dtype == "str":
            return pa.string()
        elif self.dtype == "struct":
            return pa.struct([field.to_pyarrow_schema() for field in self.fields])
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}.")

    @staticmethod
    def from_json(json_data):
        """
        Create a ColumnMeta object from a JSON object.
        """
        name = json_data["name"]
        dtype = json_data["dtype"]
        distribution = json_data.get("distribution", "normal")
        str_length = json_data.get("str_length", None)
        custom_generator = None
        num_null = json_data.get("num_null", 0)
        cardinality = json_data.get("cardinality", None)
        num_repeated = json_data.get("num_repeated", 1)
        dist_params = json_data.get("dist_params", {})
        if dtype == "struct":
            fields = [ColumnMeta.from_json(child) for child in json_data["fields"]]
        else:
            fields = None
        compression = json_data.get("compression", "NONE")
        encoding = json_data.get("encoding", "PLAIN")
        return ColumnMeta(
            name,
            dtype,
            distribution,
            str_length,
            custom_generator,
            num_null,
            cardinality,
            num_repeated,
            dist_params,
            fields,
            compression,
            encoding,
        )
