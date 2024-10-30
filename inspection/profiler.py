import cProfile
import pstats
import io
import logging

from src.processor.loader import PalXFELLoader
from src.config.config import load_config, ExpConfig
from src.filesystem import get_run_scan_directory


def main() -> None:
    """Profile program with cProfile module and visualize with tuna."""
    config: ExpConfig = load_config()
    load_dir: str = config.path.load_dir
    file: str = get_run_scan_directory(load_dir, 150, 1, 10)

    logging_file: str = 'logs/profiling/profiling.log'
    # logging Setting
    logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(message)s')

    # Create a profiler object
    profiler = cProfile.Profile()

    # Enable the profiler
    profiler.enable()

    # Run the main function from the other file
    PalXFELLoader(file)
    # processing_main.main()

    # Disable the profiler
    profiler.disable()

    # Create a Stats object and sort the results by cumulative time
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')

    # Redirect the stats output to a StringIO object
    output_stream = io.StringIO()
    stats.stream = output_stream

    # Print the stats to the StringIO object
    stats.print_stats()

    # Get the captured output
    profiling_results = output_stream.getvalue()

    # Log the captured output
    logging.info(profiling_results)

    # Save the profiling results to a file
    stats_file = 'logs/profiling/profiling.prof'
    stats.dump_stats(stats_file)

    print(f"Profiling results logged to '{logging_file}'")
    print(f"Profiling results logged to '{stats_file}'")


if __name__ == "__main__":
    main()
