"""
utility functions for loading models and running prediction
"""

import time
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as rt


# datasets

class Dataset():
    def __init__(
            self,
            examples: List[str],
            tokenizer: AutoTokenizer,
            padding: str = "max_length"):

        self.encodings = tokenizer(examples, truncation=True, padding=padding)
        self.labels = [1] * len(examples)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def read_text(text_filepath: str) -> List[str]:
    """
    read lines of text from a file
    """
    with open(text_filepath, "r") as f:
        examples = f.readlines()
    examples = [e.strip() for e in examples]
    return examples


# model / tokenizer loading

def load_tokenizer(tokenizer_name):
    """
    load tokenizer
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_model(model_name, torchscript=False):
    """
    load classification model from Huggingface.co
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torchscript=torchscript)
    # time.sleep(0.4)
    return model


def onnx_load_model(onnx_model_filepath):
    """
    load ONNX model using onnxruntime
    """
    session = rt.InferenceSession(onnx_model_filepath)
    # time.sleep(0.4)
    return session


def torchscript_load_model(torchscript_model_path: str):
    """
    load a saved torchscript model
    """
    model = torch.jit.load(torchscript_model_path)
    # time.sleep(0.4)
    model.eval()
    return model


# model prediction

def torchscript_predict(model, dataset: Dataset):
    """
    run inference using a PyTorch model and dataset
    """
    input_ids_tensor = torch.tensor(dataset.encodings["input_ids"])
    attention_masks_tensor = torch.tensor(dataset.encodings["attention_mask"])
    with torch.no_grad():  # shouldn't actually need to do this since model is in eval mode, but memory usage increases during inference if I don't
        output = model(input_ids_tensor, attention_masks_tensor)
    if type(output) == dict:
        return output["logits"].detach().numpy()
    else:
        return output[0].detach().numpy()


def onnx_predict(session, dataset: Dataset):
    """
    run inference using an onnxruntime session and dataset
    """
    label_name = session.get_outputs()[0].name
    return session.run([label_name], dict(dataset.encodings))[0]


def mxnet_predict(net, dataset: Dataset):
    """
    run inference using an MXNet model and dataset
    """
    return net(dict(dataset.encodings))[0]
