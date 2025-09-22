import os
import sys
from typing import Any
from yaml import safe_load
from src.logger import logging
from src.exception import MyException
from dill import load as dill_load, dump as dill_dump
from numpy import load as numpy_load, save as numpy_save


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
        print()
        logging.info(f"Reading YAML file...")
        with open(file=filepath, mode="rb") as f:
            data: Any = safe_load(f)

        logging.info("YAML file read successfully.")
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
        print()
        logging.info(f"Loading object...")

        with open(file=filepath, mode="rb") as f:
            obj: Any = dill_load(f)

        logging.info("Object loaded successfully.")
        return obj

    except Exception as e:
        raise MyException(e, sys) from e


def save_object(filepath: str, obj: Any) -> None:
    """
    Save a Python object to file using dill.

    Args:
        filepath (str): File path where to save the object.
        obj (Any): Python object to serialize and save.

    Raises:
        MyException: If saving fails.
    """
    try:
        print()
        logging.info(f"Saving object...")

        obj_dir: str = os.path.dirname(filepath)
        os.makedirs(obj_dir, exist_ok=True)

        with open(file=filepath, mode="wb") as f:
            dill_dump(obj=obj, file=f)

        logging.info("Object saved successfully.")

    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array(filepath: str):
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
        print()
        logging.info(f"Loading numpy array...")

        with open(file=filepath, mode="rb") as f:
            arr: Any = numpy_load(f)

        logging.info("Numpy array loaded successfully.")
        return arr

    except Exception as e:
        raise MyException(e, sys) from e


def save_numpy_array(filepath: str, np_array) -> None:
    """
    Save a NumPy array to a binary file.

    Args:
        filepath (str): Path where the .npy file will be saved.
        np_array (numpy.ndarray): NumPy array to save.

    Raises:
        MyException: If saving fails.
    """
    try:
        print()
        logging.info(f"Saving numpy array...")

        array_dir: str = os.path.dirname(filepath)
        os.makedirs(array_dir, exist_ok=True)

        with open(file=filepath, mode="wb") as f:
            numpy_save(f, np_array)

        logging.info("Numpy array saved successfully.")

    except Exception as e:
        raise MyException(e, sys) from e
