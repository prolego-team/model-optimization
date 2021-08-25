# model-optimization

Prepare and optimize a transformers-based PyTorch model for deployment.

The following options, as well as their impact on runtime and memory consumption, are explored:

- Model serialization with TorchScript or ONNX
- Model quantization with TorchScript or ONNX Runtime
- Pruning attention heads
- Different model architectures (Huggingface.co roberta-base vs. distilroberta-base)
