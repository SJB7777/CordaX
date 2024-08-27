import os
from datetime import datetime

import click
import h5py

from src.config.config import load_config
from src.filesystem import get_run_scan_directory


def list_files_in_directory(directory: str, show_size: bool, show_modified: bool, show_hdf5_keys: bool) -> None:
    """List files in the given directory, optionally showing size, modification date, and HDF5 keys information."""
    if not os.path.exists(directory):
        raise click.ClickException(f"Directory '{directory}' not found.")

    click.echo(directory)
    scan_names: list[str] = sorted(os.listdir(directory))

    for scan_name in scan_names:
        click.echo(scan_name)
        scan_dir = os.path.join(directory, scan_name)
        file_names: list[str] = sorted(os.listdir(scan_dir))

        for file_name in file_names:
            file_path = os.path.join(scan_dir, file_name)
            line = [file_name]

            if show_size:
                file_size = os.path.getsize(file_path)
                line.append(f'size {file_size} (bytes)')

            if show_modified:
                time = os.path.getmtime(file_path)
                date = datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
                line.append(f'date {date}')

            if show_hdf5_keys and file_name.endswith('.h5'):
                with h5py.File(file_path, 'r') as hdf5_file:
                    keys = list(hdf5_file.keys())
                    line.append(f'keys {", ".join(keys)}')

            click.echo(' '.join(line))


@click.command()
@click.argument('run_n', type=int)
@click.option('--size', is_flag=True, help='Show detailed size information')
@click.option('--date', is_flag=True, help='Show file modification date')
@click.option('--keys', is_flag=True, help='Show keys contained in HDF5 files')
def file_check(run_n: int, size: bool, date: bool, keys: bool) -> None:
    """List files of the run directory"""
    config = load_config()
    load_dir = config.path.load_dir
    run_dir = get_run_scan_directory(load_dir, run_n)

    list_files_in_directory(run_dir, size, date, keys)


if __name__ == '__main__':
    file_check()