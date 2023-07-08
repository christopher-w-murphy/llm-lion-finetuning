from src.domain.model import get_lora_config, get_training_arguments


def test_get_lora_config():
    result = get_lora_config()
    assert result.r == 16


def test_get_training_arguments():
    output_dir = 'foo/bar'
    n_epochs = 42
    result = get_training_arguments(output_dir, n_epochs)
    assert not result.auto_find_batch_size
