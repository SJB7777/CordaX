import os

from typing import Optional


def get_run_scan_directory(mother: str, run: int, scan: Optional[int] = None, file_num: Optional[int] = None) -> str:
    """
    Generate the directory for a given run and scan number, optionally with a file number.

    Parameters:
        mother (str): The base directory or path where the path will be generated.
        run (int): The run number for which the path will be generated.
        scan (int, optional): The scan number for which the path will be generated.
            If not provided, only the run directory path will be returned.
        file_num (int, optional): The file number for which the path will be generated.
            If provided, both run and scan directories will be included in the path.

    Returns:
        str: The path representing the specified run, scan, and file number (if applicable).
    """

    if scan is None and file_num is None:
        return os.path.join(mother, f"run={run:0>3}")
    if scan is not None and file_num is None:
        return os.path.join(mother, f"run={run:0>3}", f"scan={scan:0>3}")
    if scan is not None and file_num is not None:
        return os.path.join(mother, f"run={run:0>3}", f"scan={scan:0>3}", f"p{file_num:0>4}.h5")


def make_run_scan_directory(mother: str, run: int, scan: int) -> str:
    """
    Create a nested directory structure for the given run and scan numbers.

    Parameters:
        dir (str): The base directory where the nested structure will be created.
        run (int): The run number for which the directory will be created.
        scan (int): The scan number for which the directory will be created.

    Returns:
        str: The path of the created nested directory.
    """

    os.makedirs(mother, exist_ok=True)
    path = os.path.join(mother, f'run={run:0>3d}')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'scan={scan:0>3d}')
    os.makedirs(path, exist_ok=True)
    return path
