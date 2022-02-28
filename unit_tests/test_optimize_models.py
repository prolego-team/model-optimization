"""
unit tests for optimize_models.py
"""

from tempfile import mkdtemp
import os
import shutil

import pytest

import optimize_models


@pytest.mark.usefixtures("model_for_torchscript")
@pytest.mark.usefixtures("tokenizer")
def test_torchscript_trace(model_for_torchscript, tokenizer):
    """
    """
    torchscript_model = optimize_models.torchscript_trace(
        model_for_torchscript, tokenizer)
    assert torchscript_model is not None


@pytest.mark.usefixtures("model_name")
def test_quantize_pytorch_model(model_name: str):
    """
    test that the traced pytorch model is created in the output dir
    """
    output_dirpath = mkdtemp(dir="./")
    optimize_models.quantize_pytorch_model(
        model_name, model_name, output_dirpath)
    assert os.path.exists(os.path.join(output_dirpath, "model.pt"))
    shutil.rmtree(output_dirpath)


@pytest.mark.usefixtures("model_name")
def test_to_onnx(model_name: str):
    """
    test that a model called model.onnx is created in the output dir
    """
    output_dirpath = mkdtemp(dir="./")
    optimize_models.to_onnx(model_name, output_dirpath)
    assert os.path.exists(os.path.join(output_dirpath, "model.onnx"))
    shutil.rmtree(output_dirpath)
