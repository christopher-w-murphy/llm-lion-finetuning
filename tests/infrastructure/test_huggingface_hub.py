from datetime import datetime

from src.infrastructure.huggingface_hub import format_log_filename, dicttobytes


def test_format_log_filename():
    test_timestamp = datetime(2022, 7, 30, 1, 2, 3)
    result = format_log_filename(test_timestamp)
    assert result == 'log_2022-07-30_01:02:03.json'


def test_dicttobytes():
    test_dict = {'foo': 123, 'bar': True}
    result = dicttobytes(test_dict)
    assert result.getvalue() == b'{"foo": 123, "bar": true}'
