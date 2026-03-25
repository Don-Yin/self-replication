"""self-replication phase diagram experiment."""
from pathlib import Path
from pyinstrument import Profiler

profiler = Profiler()


def start_profiling():
    """start the pyinstrument profiler."""
    profiler.start()


def stop_profiling(output_path=None):
    """stop profiler and print or save the report."""
    profiler.stop()
    if output_path:
        Path(output_path).write_text(profiler.output_text(unicode=True))
    else:
        profiler.print()
