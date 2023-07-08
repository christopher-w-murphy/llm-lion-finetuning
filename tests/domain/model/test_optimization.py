from src.domain.model.optimization import adamw_hyperparameters, lion_hyperparameters


def test_hyperparameters():
    assert len(adamw_hyperparameters) == len(lion_hyperparameters)
    assert len(adamw_hyperparameters) == 2
