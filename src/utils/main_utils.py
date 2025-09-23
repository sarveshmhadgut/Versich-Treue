import os
import sys
from typing import Any
from yaml import safe_load
from src.logger import logging
from src.exception import MyException
from pandas import read_csv, DataFrame
from dill import load as dill_load, dump as dill_dump
from numpy import load as numpy_load, save as numpy_save



def read_csv_file(filepath: str) -> DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        MyException: If reading the CSV file fails.
    """
    try:
        data = read_csv(filepath)
        return data

    except Exception as e:
        raise MyException(e, sys) from e


def save_df_as_csv(df: DataFrame, filepath: str, **kwargs) -> None:
    """
    Save a pandas DataFrame as a CSV file.

    Args:
        df (DataFrame): DataFrame to save.
        filepath (str): Location where the CSV will be saved.
        **kwargs: Additional keyword arguments for pandas to_csv().

    Raises:
        MyException: If saving the DataFrame fails.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, **kwargs)

    except Exception as e:
        raise MyException(e, sys) from e


def read_yaml_file(filepath: str) -> Any:
    """
    Read and parse a YAML file safely.

    Args:
        filepath (str): Full path to the YAML file.

    Returns:
        Any: Parsed data from YAML file.

    Raises:
        MyException: If reading or parsing fails.
    """
    try:
        with open(filepath, "r") as f:
            data = safe_load(f)
        return data

    except Exception as e:
        raise MyException(e, sys) from e


def load_object(filepath: str) -> Any:
    """
    Load a Python object using dill from a file.

    Args:
        filepath (str): File path to load the object from.

    Returns:
        Any: The loaded Python object.

    Raises:
        MyException: If loading fails.
    """
    try:
        with open(filepath, "rb") as f:
            obj = dill_load(f)
        return obj

    except Exception as e:
        raise MyException(e, sys) from e


def save_object(obj: Any, filepath: str) -> None:
    """
    Save a Python object to file using dill.

    Args:
        obj (Any): Python object to serialize and save.
        filepath (str): File path where to save the object.

    Raises:
        MyException: If saving fails.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            dill_dump(obj, f)

    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise MyException(e, sys) from e


def load_numpy_array(filepath: str) -> Any:
    """
    Load a NumPy array from a binary file.

    Args:
        filepath (str): Path to the .npy binary file.

    Returns:
        numpy.ndarray: Loaded NumPy array.

    Raises:
        MyException: If loading fails.
    """
    try:
        with open(filepath, "rb") as f:
            arr = numpy_load(f)

        return arr

    except Exception as e:
        logging.error(f"Error loading numpy array: {e}")
        raise MyException(e, sys) from e


def save_numpy_array(np_array: Any, filepath: str) -> None:
    """
    Save a NumPy array to a binary file.

    Args:
        np_array (numpy.ndarray): NumPy array to save.
        filepath (str): Path where the .npy file will be saved.

    Raises:
        MyException: If saving fails.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            numpy_save(f, np_array)

    except Exception as e:
        raise MyException(e, sys) from e
