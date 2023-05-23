import pytest

from src.domain.configuration import get_base_model_id, get_tokenizer_id, get_output_dir


def test_get_base_model_id():
    result = get_base_model_id('XXL')
    assert result == "philschmid/flan-t5-xxl-sharded-fp16"
    with pytest.raises(ValueError):
        get_base_model_id('large')


def test_get_tokenizer_id():
    result = get_tokenizer_id('Small')
    assert result == "google/flan-t5-small"
    with pytest.raises(ValueError):
        get_tokenizer_id('large')


def test_get_output_dir():
    result = get_output_dir('Foo', 'Bar')
    assert result == 'lora-flan-t5-foo-bar'
