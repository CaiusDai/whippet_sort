import numpy as np
import string
import matplotlib.pyplot as plt
import scipy.stats as stats

def generate_column_data(meta, size):
    if meta.custom_generator is not None:
        return meta.custom_generator()
    data_size = size- meta.num_null
    data = None
    if meta.dtype == "int":
        data = generate_int_data(meta, data_size)
    elif meta.dtype == "double":
        data = generate_double_data(meta, data_size)
    elif meta.dtype == "str":
        data = generate_str_data(meta, data_size)
    elif meta.dtype == "struct":
        data = generate_struct_data(meta, data_size)
    else:
        raise ValueError(f"Unsupported dtype: {meta.dtype}.")
    # Append null values
    data.append([None] * meta.num_null)
    return data

def generate_int_data(meta, size):
    return generate_numeric_data(meta, size, is_int=True)


def generate_double_data(meta, size):
    return generate_numeric_data(meta, size)


def generate_str_data(meta, size):
    if meta.dtype != "str":
        raise ValueError(
            f"Unsupported dtype: {meta.dtype} for function generate_str_data."
        )
    # Read in essential configurations
    distribution = meta.distribution
    dist_config = meta.dist_params
    str_length = meta.str_length
    if str_length is None:
        str_length = (1, 10)
    if str_length[0] > str_length[1]:
        raise ValueError(f"Invalid string length range: {str_length}.")
    cardinality = meta.cardinality

    if distribution == "uniform":
        if cardinality is None:
            print(
                "[Warning] Cardinality is not set for column {meta.name}. The cardinality will be randomly selected."
            )
            cardinality = np.random.randint(1, size)
        selection_pool = generate_str_array(
            cardinality, str_length[0], str_length[1], unique=True
        )
        # Numpy choice function use uniform distribution by default
        return np.random.choice(selection_pool, size)
    elif distribution == "unique":
        if cardinality is not None and cardinality != size:
            print(
                f"[Warning] For column {meta.name}, cardinality {cardinality} is not equal to the size {size}, and will be ignored."
            )
        return generate_str_array(size, str_length[0], str_length[1], unique=True)
    elif distribution == "random":
        if cardinality is None:
            return generate_str_array(size, str_length[0], str_length[1])
        else:
            selection_pool = generate_str_array(
                cardinality, str_length[0], str_length[1], unique=True
            )
            return np.random.choice(selection_pool, size)
    elif distribution == "normal":
        std = dist_config.get("std", 1)
        mean = 0
        if cardinality is None:
            cardinality = np.random.randint(1, size)
        data = generate_str_array(size, str_length[0], str_length[1],unique=True)
        pdf = stats.norm.pdf(np.linspace(-(cardinality/2), cardinality/2, str_length[1] - str_length[0]),mean,std)
        probilities = pdf / probilities.sum()
        return np.random.choice(data, size, p=probilities)
    elif distribution == "gamma":
        shape = dist_config.get("shape", 2.0)
        scale = dist_config.get("scale", 1.0)
        if cardinality is None:
            cardinality = np.random.randint(1, size)
        data = generate_str_array(size, str_length[0], str_length[1],unique=True)
        pdf = stats.gamma.pdf(np.linspace(-(cardinality/2), cardinality/2, str_length[1] - str_length[0]),shape,scale)
        probilities = pdf / probilities.sum()
        return np.random.choice(data, size, p=probilities)
    else:
        raise ValueError(
            f"Unsupported distribution: {distribution} for function generate_str_data."
        )

def generate_struct_data(meta, size):
    if meta.dtype != "struct":
        raise ValueError(
            f"Unsupported dtype: {meta.dtype} for function generate_struct_data."
        )
    fields = meta.fields
    data = {}
    for field in fields:
        data[field.name] = generate_column_data(field, size)
    return data


def generate_numeric_data(meta, size, is_int=False):
    # Check if the type is int or double
    if meta.dtype != "int" and meta.dtype != "double":
        raise ValueError(
            f"Unsupported dtype: {meta.dtype} for function generate_numeric_data."
        )
    # Read essential configurations
    distribution = meta.distribution
    dist_config = meta.dist_params
    min_val = dist_config.get("min", np.iinfo(np.int32).min)
    max_val = dist_config.get("max", np.iinfo(np.int32).max)
    cardinality = meta.cardinality
    data = None

    # Generate data based on the distribution
    if distribution == "normal":
        mean = dist_config.get("mean", 0)
        std = dist_config.get("std", 1)
        data = np.random.normal(mean, std, size)
    elif distribution == "uniform":
        data = np.random.uniform(min_val, max_val, size)
    elif distribution == "unique":
        if cardinality is not None and cardinality != size:
            print(
                f"[Warning] For column {meta.name}, cardinality {cardinality} is not equal to the size {size}, and will be ignored."
            )
        data = np.random.choice(np.arange(min_val, max_val), size, replace=False)
    elif distribution == "gamma":
        shape = dist_config.get("shape", 2.0)
        scale = dist_config.get("scale", 1.0)
        data = np.random.gamma(shape, scale, size)
    elif distribution == "random":
        data = np.random.rand(size)
    else:
        raise ValueError(
            f"Unsupported distribution: {distribution} for function generate_numeric_data."
        )

    # Mapping to desired range and cardinality
    plot_value_counts(data)  # Debug
    data = map_to_range(data, min_val, max_val, cardinality)
    if is_int:
        data = data.astype(np.int32)
    return data


def map_to_range(data, low=None, high=None, cardinality=None):
    """
    Map the data to the given range with specified cardinality.
    - Data: The input data to be mapped. Must be numeric.
    - low: The lower bound of the range. Inclusive. If not set, the minimum value of the data will be used.
    - high: The upper bound of the range. Inclusive. If not set, the maximum value of the data will be used.
    - cardinality: The number of unique values in the mapped data. If not set, the data will be mapped to the given range without changing the cardinality.
    """
    # Check data type must be numeric
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(
            f"map_to_range only supports numeric data type. Given Data type: {data.dtype}."
        )
    # Step 1: Map data to the given range
    if low is None:
        low = np.min(data)
    if high is None:
        high = np.max(data)
    mapped_data = (data - min(data)) / (max(data) - min(data)) * (high - low) + low

    print("Mapped data")
    plot_value_counts(mapped_data)
    # Step 2: Check cardinality
    if cardinality is None:
        return mapped_data
    elif cardinality > len(np.unique(mapped_data)):
        raise ValueError(
            f"Cardinality {cardinality} is greater than the number of unique values {len(np.unique(mapped_data))}."
        )
    else:
        # Step 3: Map the data to the given cardinality
        assign_bin = np.linspace(low, high, cardinality)
        dividing_bin = np.linspace(low, high, cardinality + 1)
        inds = np.digitize(mapped_data, dividing_bin)
        print("Index")
        plot_value_counts(inds)
        # Change indexs that are equal to cardinality to cardinality-1
        inds[inds == cardinality + 1] = cardinality
        return assign_bin[inds - 1]


def generate_str_array(size, len_min, len_max, unique=False):
    if len_min > len_max:
        raise ValueError("len_min must be less than or equal to len_max")
    if size < 1:
        raise ValueError("array_size must be at least 1")

    unique_strings = set()
    array = []
    # Split the string into individual characters
    characters = list(string.ascii_letters + string.digits)
    while len(array) < size:
        length = np.random.randint(len_min, len_max)
        # Generate a string of the determined length
        random_string = "".join(np.random.choice(characters, length))

        # Ensure the string is unique before adding
        if unique and random_string not in unique_strings:
            unique_strings.add(random_string)
            array.append(random_string)
        elif not unique:
            array.append(random_string)

    return array

def plot_value_counts(arr):
    """
    Plots a graph of unique values (x-axis) vs their counts (y-axis) from the given array.

    Parameters:
    - arr (array-like): An array of values to be plotted.
    """
    # Count the occurrence of each unique value
    values, counts = np.unique(arr, return_counts=True)
    print(counts)
    plt.figure(figsize=(10, 6))  
    plt.plot(values, counts, marker="o")  
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Value Counts")
    plt.grid(axis='y', linestyle='--')  # Add horizontal grid lines for better readability
    plt.show()
