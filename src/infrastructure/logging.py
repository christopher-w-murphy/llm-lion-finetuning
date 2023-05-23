from datetime import datetime
from logging import INFO, Formatter, FileHandler
from pathlib import Path


def get_logging_dir() -> Path:
    log_dir = Path.cwd() / 'logs'
    log_dir.mkdir(exist_ok=True)
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return log_dir / f'logs_{start_time}.txt'


def get_formatter() -> Formatter:
    return Formatter('%(asctime)s | %(levelname)s | %(message)s')


def get_file_handler(logging_dir: Path) -> FileHandler:
    file_handler = FileHandler(logging_dir)
    file_handler.setLevel(INFO)
    file_handler.setFormatter(get_formatter())
    return file_handler
