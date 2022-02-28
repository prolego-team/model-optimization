"""
utility functions to optimize a trained transformers model through:
1. serialization (ONNX or TorchScript tracing)
2. dynamic quantization (PyTorch/TorchScript or ONNX Runtime)
3. pruning random heads

Note: pruning random heads is only intended to be used an an experimental method to
measure the impact of pruning on resource usage. The correct way to prune attention
heads should be based on the relative importance of each head (see https://arxiv.org/abs/1905.10650)
"""

import os
import warnings

import torch
from torch.onnx import export
from onnxruntime.quantization import quantize_dynamic, QuantType

from model_utils import load_model, load_tokenizer


# model serialization

def to_onnx(model_name: str, output_dirpath: str, feature_name: str = "sequence-classification") -> None:
    """
    Convert a transformers model to ONNX and save at output_filepath.
    From: https://huggingface.co/docs/transformers/master/serialization
    Note: Assumes the tokenizer is saved alongside the model (i.e., tokenizer name
    matches the model name).
    """
    command = "python -m transformers.onnx"
    command += " --model=" + model_name
    command += " --feature=" + feature_name
    command += " " + output_dirpath
    os.system(command)


def torchscript_trace(model, tokenizer):
    """
    trace a transformers model using torch.jit.trace
    """
    model.eval()
    inputs = tokenizer.encode_plus(
        "sample text",
        return_tensors = 'pt')
    with torch.no_grad():
        # strict=False needed for quantization
        torchscript_model = torch.jit.trace(model, [inputs["input_ids"], inputs["attention_mask"]], strict=False)
    return torchscript_model


def torchscript_save(torchscript_model, output_filepath):
    """
    save a traced model to output_filepath
    """
    torch.jit.save(torchscript_model, output_filepath)


def to_torchscript(model_name, tokenizer_name, output_dirpath, method: str = "tracing"):
    """
    convert a transformers model to TorchScript and save to output_dirpath
    method is either "tracing" or "scripting", but scripting is currently not supported
    """
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
    # method is either 'tracing' or 'scripting' (not supported)
    model = load_model(model_name, torchscript=True)
    tokenizer = load_tokenizer(tokenizer_name)
    if method == "tracing":
        torchscript_model = torchscript_trace(model, tokenizer)
    else:
        warnings.warn("'scripting' is not currently supported, please use 'tracing'")
        model.eval()
        torchscript_model = torch.jit.script(model)
    torchscript_save(torchscript_model, os.path.join(output_dirpath, method + ".pt"))


# model quantization

def quantize_onnx_model(onnx_model_path: str, output_filepath: str) -> None:
    """quantize an ONNX model using onnxruntime and save to output_filepath"""
    quantize_dynamic(onnx_model_path, output_filepath, weight_type=QuantType.QInt8)


def quantize_pytorch_model(model_name: str, tokenizer_name: str, output_dirpath: str) -> None:
    """
    quantize linear layers of a PyTorch model and serialize/save to output_dirpath using
    TorchScript
    """

    model = load_model(model_name)
    model_int8 = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    tokenizer = load_tokenizer(tokenizer_name)
    torchscript_model = torchscript_trace(model_int8, tokenizer)
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
    torchscript_save(torchscript_model, os.path.join(output_dirpath, "model.pt"))


# pruning

def prune_random_heads(model_name, frac_heads_per_layer, output_dir):
    """
    randomly prune fractional number of attention heads per layer from a model and save to output_dir

    Note: pruning random heads is only intended to be used an an experimental method to
    measure the impact of pruning on resource usage. The correct way to prune attention
    heads should be based on the relative importance of each head (see https://arxiv.org/abs/1905.10650)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    heads_to_prune = list(range(int(n_heads * frac_heads_per_layer)))
    heads_to_prune = {layer: heads_to_prune for layer in range(n_layers)}
    model.prune_heads(heads_to_prune)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
