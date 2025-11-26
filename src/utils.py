"""
This file must be placed in 'src' directory
"""

from contextlib import contextmanager
from datetime import timedelta
import os
import pathlib
import pickle
import time


def get_project_root() -> pathlib.Path:
    """
    Determines the project root using a fixed relative path from this file.
    It assumes this file is located within the 'project_root/src' directory.
    Validates the assumed root by checking for an 'src' directory within it.

    Returns:
        pathlib.Path: The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the 'src' directory is not found at the
                           assumed project root, indicating a potential
                           misconfiguration or change in directory structure.
        RuntimeError: If this utility file's location has changed such that
                      the fixed relative path logic is no longer valid.
    """
    try:
        # Get the absolute path of the current file (path_helpers.py)
        this_file_path = pathlib.Path(__file__).resolve()

        # Assumed structure: project_root/src/<this file>.py
        assumed_project_root = this_file_path.parent.parent
    except IndexError:
        # This would happen if Path(__file__).parent goes above filesystem root
        # which implies the file is not deep enough for this logic.
        raise RuntimeError(
            f"The utility file '{__file__}' seems to be located too high in the"
            f"directory tree for the fixed relative path logic to apply."
            f"Expected 'project_root/src/<this file>.py'."
            f"Instead got {this_file_path}."
        )

    # Validate: Check for the presence of an 'src' directory in the assumed root.
    # This 'src' directory is the one directly under the project_root.
    expected_src_dir = assumed_project_root / "src"

    if not (expected_src_dir.exists() and expected_src_dir.is_dir()):
        raise FileNotFoundError(
            f"Validation failed: An 'src' directory was not found at the assumed "
            f"project root '{assumed_project_root}'.\n"
            f"This function expects the project structure to be 'project_root/src/...', "
            f"and this utility file ('{__file__}') to be at a certain fixed location "
            f"within 'src/'. If the structure or file location has changed, "
            f"this function may need an update."
        )

    return assumed_project_root


# Method to pickle and save an object to a file
def save(obj, filename: str, exists_ok=True):
    """Pickles and saves the object to a file.

    Args:
        obj: The object to pickle and save.
        filename (str): The file path to save the object.
        exists_ok (bool): If False, raises an exception if the file already exists.
    """
    if not exists_ok and os.path.exists(filename):
        raise FileExistsError(f"File '{filename}' already exists.")
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


# Method to unpickle and load an object from a file
def load(filename: str):
    """Unpickles and loads the object from a file.

    Args:
        filename (str): The file path to load the object from.
    Returns:
        The loaded object.
    """
    with open(filename, "rb") as file:
        obj = pickle.load(file)
    return obj


@contextmanager
def timer(message:bool ="Time taken: {}", disable:bool = False):
    start = time.perf_counter()
    yield
    elapsed = timedelta(seconds=time.perf_counter() - start)
    if not disable:
        print(message.format(elapsed))