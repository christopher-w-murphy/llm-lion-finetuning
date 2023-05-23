from io import StringIO
from logging import INFO, Formatter, StreamHandler


def get_formatter() -> Formatter:
    return Formatter('%(asctime)s | %(levelname)s | %(message)s')


def get_stream_handler(io: StringIO) -> StreamHandler:
    stream_handler = StreamHandler(io)
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(get_formatter())
    return stream_handler
