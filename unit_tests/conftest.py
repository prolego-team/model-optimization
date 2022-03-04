import pytest

from model_utils import load_model, load_tokenizer


@pytest.fixture
def model_name() -> str:
    return "roberta-base"


@pytest.fixture
def tokenizer(model_name: str):
    return load_tokenizer(model_name)


@pytest.fixture
def model(model_name):
    return load_model(model_name, torchscript=False)


@pytest.fixture
def model_for_torchscript(model_name):
    return load_model(model_name, torchscript=True)