"""
utility functions to optimize a trained transformers model through:
1. serialization (ONNX or TorchScript)
2. quantization (PyTorch/TorchScript or ONNX Runtime)
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

def ensure_valid_input(model, tokens, input_names):
    """
    Ensure input are presented in the correct order, without any Non
    Args:
        model: The model used to forward the input data
        tokens: BatchEncoding holding the input data
        input_names: The name of the inputs
    Returns: Tuple

    Used for conversion from PyTorch to ONNX model
    Copied from https://github.com/huggingface/transformers/blob/af8afdc88dcb07261acf70aee75f2ad00a4208a4/src/transformers/convert_graph_to_onnx.py
    """
    print("Ensuring inputs are in correct order")

    model_args_name = model.forward.__code__.co_varnames
    model_args, ordered_input_names = [], []
    for arg_name in model_args_name[1:]:  # start at index 1 to skip "self" argument
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            print(f"{arg_name} is not present in the generated input list.")
            break

    print("Generated inputs order: {}".format(ordered_input_names))
    return ordered_input_names, tuple(model_args)


def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
    """
    Used for conversion from PyTorch to ONNX model.
    Copied from https://github.com/huggingface/transformers/blob/af8afdc88dcb07261acf70aee75f2ad00a4208a4/src/transformers/convert_graph_to_onnx.py
    """
    if isinstance(tensor, (tuple, list)):
        return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]

    else:
        # Let's assume batch is the first axis with only 1 element (~~ might not be always true ...)
        axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == 1][0]: "batch"}
        if is_input:
            if len(tensor.shape) == 2:
                axes[1] = "sequence"
            else:
                raise ValueError(f"Unable to infer tensor axes ({len(tensor.shape)})")
        else:
            seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
            axes.update({dim: "sequence" for dim in seq_axes})

    print(f"Found {'input' if is_input else 'output'} {name} with shape: {axes}")
    return axes


def infer_shapes(model, tokenizer):
    """
    Uses for conversion from PyTorch to ONNX model.
    Copied from https://github.com/huggingface/transformers/blob/af8afdc88dcb07261acf70aee75f2ad00a4208a4/src/transformers/convert_graph_to_onnx.py
    """
    tokens = tokenizer("sample text", return_tensors="pt")
    seq_len = tokens.input_ids.shape[-1]
    outputs = model(**tokens).to_tuple()
    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: build_shape_dict(k, v, True, seq_len) for k, v in tokens.items()}

    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)

    output_names = [f"output_{i}" for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for k, v, in zip(output_names, outputs_flat)}

    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_vars, output_names, dynamic_axes, tokens


def to_onnx(model_name: str, tokenizer_name: str, output_filepath: str) -> None:
    """
    Convert a transformers model to ONNX and save at output_filepath.
    Modified from https://github.com/huggingface/transformers/blob/af8afdc88dcb07261acf70aee75f2ad00a4208a4/src/transformers/convert_graph_to_onnx.py
    """

    output_dirpath, _ = os.path.split(output_filepath)
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    model = load_model(model_name)
    tokenizer = load_tokenizer(tokenizer_name)
    model.eval()
    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(model, tokenizer)
        ordered_input_names, model_args = ensure_valid_input(model, tokens, input_names)
        export(
            model,
            model_args,
            f=output_filepath,
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            enable_onnx_checker=True,
            opset_version=11)


def torchscript_trace(model, tokenizer):
    """
    trace a transformers model using torch.jit.trace
    """
    inputs = tokenizer.encode_plus(
        "sample text",
        max_length = 128,
        pad_to_max_length=True,
        add_special_tokens=True,
        return_tensors = 'pt')
    with torch.no_grad():
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
    model.eval()
    tokenizer = load_tokenizer(tokenizer_name)
    if method == "tracing":
        torchscript_model = torchscript_trace(model, tokenizer)
    else:
        warnings.warn("'scripting' is not currently supported, please use 'tracing'")
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
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    heads_to_prune = list(range(int(n_heads * frac_heads_per_layer)))
    heads_to_prune = {layer: heads_to_prune for layer in range(n_layers)}
    model.prune_heads(heads_to_prune)
    model.save_pretrained(output_dir)
