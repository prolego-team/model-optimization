# model-optimization

Prepare and optimize a transformers-based PyTorch model for deployment.

The following options, as well as their impact on runtime and memory consumption, are explored:

- Model serialization with TorchScript or ONNX
- Model quantization with TorchScript or ONNX Runtime
- Pruning attention heads
- Different model architectures (Huggingface.co roberta-base vs. distilroberta-base)

## Getting Started

Use pyenv(https://github.com/pyenv/pyenv#installation) to install python v. 3.9.2:

    pyenv install 3.9.2

Use poetry(https://python-poetry.org/docs/) to create the environment and install dependencies:

    cd model-optimization
    poetry install
