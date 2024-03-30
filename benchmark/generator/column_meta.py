#!/usr/bin/env python3
import numpy as np
import pyarrow as pa


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
        compression="NONE",
        encoding="PLAIN",
    ):
        """
        Initialize the ColumnMeta object with column metadata.

        Parameters:
        - name (str): The name of the column.
        - dtype (type or str): The data type of the column (e.g., int, str)
        - distribution (str): The name of the distribution to use for generating data. Not applicable for nested structures.
        - str_length (int): The length of the string. Only applicable for dtype 'str'.
        - custom_generator (callable): A custom function to generate column data. Takes precedence over distribution.
        - num_null(int): The number of null values in the column.
        - cardinality (int): The number of unique values in the column.
        - num_repeated(int): The number of times to repeat the column data.(Only used for nested structures)
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
        if encoding == None or not ColumnMeta.is_valid_encoding(dtype, encoding):
            raise ValueError(f"Unsupported encoding: {encoding} for dtype: {dtype}.")
        if compression not in ["NONE", "GZIP", "SNAPPY", "LZ4", "BROTLI"]:
            raise ValueError(f"Unsupported compression: {compression}.")
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
        field_type = self.__map_to_pyarrow_type()
        nullable = self.num_null > 0
        field = pa.field(
            self.name,
            field_type,
            nullable=nullable,
        )
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
            return pa.list_(pa.field(self.name, self.__map_to_pyarrow_type()))
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

    def is_valid_encoding(dtype, encoding):
        """
        Check if the encoding is valid for the given type.
        """
        if dtype == "int":
            return encoding in ["PLAIN", "DELTA_BINARY_PACKED", "PLAIN_DICTIONARY"]
        elif dtype == "double":
            return encoding in ["PLAIN", "BYTE_STREAM_SPLIT", "PLAIN_DICTIONARY"]
        elif dtype == "str":
            return encoding in [
                "PLAIN",
                "DELTA_BYTE_ARRAY",
                "DELTA_LENGTH_BYTE_ARRAY",
                "PLAIN_DICTIONARY",
            ]
        else:
            return False

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

        if not ColumnMeta.is_valid_encoding(dtype, encoding):
            raise ValueError(f"Unsupported encoding: {encoding} for dtype: {dtype}.")

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
