import string

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def generate_column_data(meta, size):
    if meta.custom_generator is not None:
        return meta.custom_generator()

    data_size = size - meta.num_null
    data = None
    cardinality = meta.cardinality
    # Input check
    if data_size < 0:
        raise ValueError("Size must be greater than or equal to num_null.")
    if cardinality is not None and cardinality > data_size:
        raise ValueError(
            "Can not generate data with specified cardinality, num_null and size."
        )

    # If cardinality is not set, the cardinality will be 10% of the data_size
    if cardinality is None:
        if meta.distribution == "unique" or data_size < 10:
            cardinality = data_size
        else:
            cardinality = data_size // 10

    # Generate unique data values
    if meta.dtype == "int" or meta.dtype == "double":
        min_val = meta.dist_params.get("min", np.iinfo(np.int32).min)
        max_val = meta.dist_params.get("max", np.iinfo(np.int32).max)
        data = generate_unique_num(
            cardinality, min_val, max_val, is_integer=(meta.dtype == "int")
        )
    elif meta.dtype == "str":
        str_length = meta.str_length
        if str_length is None:
            str_length = [5, 10]
        data = generate_unique_str(cardinality, str_length[0], str_length[1])
    elif meta.dtype == "struct":
        data = generate_struct_data(meta, data_size)
    else:
        raise ValueError(f"Unsupported dtype: {meta.dtype}.")

    # Apply distribution on the unique values
    if meta.distribution == "uniform":
        data = make_uniform(data, data_size)
    elif meta.distribution == "normal":
        std = meta.dist_params.get("std", 1)
        data = make_normal(data, data_size, std)
    elif meta.distribution == "gamma":
        shape = meta.dist_params.get("shape", 2.0)
        scale = meta.dist_params.get("scale", 1.0)
        data = make_gamma(data, data_size, shape, scale)

    # Append null values
    if meta.num_null > 0:
        data = np.append(data, [None] * meta.num_null)

    # Shuffle the data
    np.random.shuffle(data)
    return data


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


def generate_unique_num(size, min_val, max_val, is_integer=False):
    """
    Generate an array of unique numbers within the given range.
    - size: The number of unique numbers to generate.
    - min_val: The lower bound of the range. Inclusive.
    - max_val: The upper bound of the range. Inclusive.
    - is_integer: If True, the generated numbers will be integers. Otherwise, they will be floating-point numbers.
    """
    if is_integer:
        if (max_val - min_val + 1) < size:
            raise ValueError(
                f"Cannot generate {size} unique integers within the range [{min_val}, {max_val}]."
            )
        return np.random.choice(range(min_val, max_val + 1), size, replace=False)
    else:
        unique_floats = set()
        while len(unique_floats) < size:
            unique_floats.add(np.random.uniform(min_val, max_val))
        return np.array(list(unique_floats))


def generate_unique_str(size, str_length_min, str_length_max):
    """
    Generate an array of unique strings with random lengths within the given range.
    - size: The number of unique strings to generate.
    - str_length_min: The minimum length of the strings. Inclusive.
    - str_length_max: The maximum length of the strings. Inclusive.
    """
    if str_length_min > str_length_max:
        raise ValueError("str_length_min must be less than or equal to str_length_max")
    unique_strs = set()
    characters = list(string.ascii_letters + string.digits)
    while len(unique_strs) < size:
        length = np.random.randint(
            str_length_min, str_length_max + 1
        )  # Max is exclusive
        random_str = "".join(np.random.choice(characters, length))
        unique_strs.add(random_str)
    return np.array(list(unique_strs))


def make_uniform(values, size):
    """
    Given a list of unique values, return a list of size values sampled uniformly from the input list.
    - values: A list of unique values. If values are not unique, the output distribution is undefined.
    - size: The number of values to sample. This is the size for the return list.
    """
    if len(values) == 0:
        raise ValueError("values must have at least one element.")
    if len(values) > size:
        raise ValueError(
            "size must be greater than or equal to the number of values of input."
        )

    result = []
    freq = size // len(values)
    rest_count = size % len(values)

    # Add freq number of each value into result
    for val in values:
        result.extend([val] * freq)

    # Add the remaining values
    if rest_count > 0:
        remaining_values = np.random.choice(values, rest_count, replace=False)
        result.extend(remaining_values)
    return np.array(result)


def make_normal(values, size, std=1):
    """
    Given a list of unique values, return a list of size values sampled from a normal distribution with the given standard deviation.
    - values: A list of unique values. If values are not unique, the output distribution is undefined.
    - std: The standard deviation of the normal distribution.
    - size: The number of values to sample. This is the size for the return list.
    """
    if len(values) == 0:
        raise ValueError("values must have at least one element.")
    if len(values) > size:
        raise ValueError(
            "size must be greater than or equal to the number of values of input."
        )

    result = []
    result.extend(values)
    rest_size = size - len(values)
    if rest_size == 0:
        return np.array(result)
    cardinality = len(values)
    # Calculate the pdf for the indexes
    pdf = stats.norm.pdf(np.linspace(-2 * std, 2 * std, cardinality), 0, std)
    # Probability normalization
    probabilities = pdf / pdf.sum()
    # Add the remaining values, need to sort values first
    values = np.sort(values)
    remaining_values = np.random.choice(values, rest_size, p=probabilities)
    result.extend(remaining_values)
    return np.array(result)


def make_gamma(values, size, shape=2.0, scale=1.0):
    """
    Given a list of unique values, return a list of size values sampled from a gamma distribution with the given shape and scale.
    - values: A list of unique values. If values are not unique, the output distribution is undefined.
    - shape: The shape of the gamma distribution. Default is 2.0.
    - scale: The scale of the gamma distribution. Default is 1.0.
    - size: The number of values to sample. This is the size for the return list.
    """
    if len(values) == 0:
        raise ValueError("values must have at least one element.")
    if len(values) > size:
        raise ValueError(
            "size must be greater than or equal to the number of values of input."
        )
    result = []
    result.extend(values)
    rest_size = size - len(values)
    if rest_size == 0:
        return np.array(result)
    cardinality = len(values)
    # Calculate the pdf for the indexes, indexes is picked from 2.5 percentile to 97.5 percentile
    lower_bound = stats.gamma.ppf(0.025, shape, scale)
    upper_bound = stats.gamma.ppf(0.975, shape, scale)
    pdf = stats.gamma.pdf(
        np.linspace(lower_bound, upper_bound, cardinality), shape, scale
    )
    # Probability normalization
    probabilities = pdf / pdf.sum()
    # Add the remaining values, need to sort values first
    values = np.sort(values)
    remaining_values = np.random.choice(values, rest_size, p=probabilities)
    result.extend(remaining_values)
    return np.array(result)


def plot_value_counts(arr):
    """
    Plots a graph of unique values (x-axis) vs their counts (y-axis) from the given array.

    Parameters:
    - arr (array-like): An array of values to be plotted.
    """
    # Count the occurrence of each unique value
    arr = np.array(arr)
    values = arr[arr != None]
    print("Done")
    values, counts = np.unique(values, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.plot(values, counts, marker="o")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Value Counts")
    plt.grid(axis="y", linestyle="--")
    plt.show()
