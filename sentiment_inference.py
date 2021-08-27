"""
Run sentiment analysis (3-class classification) and report accuracy

Supports ONNX, TorchScript, and HuggingFace models and is intended to be
used to test resource use and accuracy of different model compression
methods
"""

from typing import List
from tempfile import mkdtemp
import shutil

import click
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import confusion_matrix

import model_utils
# from memory_profiler import profile # NOTE: need to comment this out to use mprof plotting


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


@profile
def tokenize(texts, tokenizer):
    return model_utils.Dataset(texts, tokenizer, padding=True)

@profile
def load_model(model_name):
    return model_utils.load_model(model_name)

@profile
def onnx_load_model(model_name):
    return model_utils.onnx_load_model(model_name)

@profile
def torchscript_load_model(model_name):
    return model_utils.torchscript_load_model(model_name)

@profile
def predict(trainer, dataset):
    return trainer.predict(test_dataset=dataset).predictions

@profile
def onnx_predict(session, dataset):
    return model_utils.onnx_predict(session, dataset)

@profile
def torchscript_predict(model, dataset):
    return model_utils.torchscript_predict(model, dataset)


@click.command()
@click.argument("model_name")
def main(model_name: str):
    tmp_dir = mkdtemp()

    # model_name = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment"

    texts = model_utils.read_text("example_data/sentiment_text.txt")
    true_labels = model_utils.read_text("example_data/sentiment_labels.txt")

    n_samples = 100
    texts = texts[:n_samples]
    true_labels = true_labels[:n_samples]
    texts = [preprocess(t) for t in texts]
    true_labels = np.array(true_labels).astype(int)

    # tokenize
    tokenizer = model_utils.load_tokenizer(tokenizer_name)
    dataset = tokenize(texts, tokenizer)

    if model_name.endswith("onnx"):
        session = onnx_load_model(model_name)
        predictions = onnx_predict(session, dataset)

    elif model_name.endswith("pt"):
        model = torchscript_load_model(model_name)
        predictions = torchscript_predict(model, dataset)

    else:
        # HuggingFace Trainer
        model = load_model(model_name)

        # create trainer
        training_args = TrainingArguments(
            output_dir=tmp_dir,
            do_train=False,
            do_eval=False,
            do_predict=True
        )
        trainer = Trainer(
            model=model,
            args=training_args
        )

        # run predict
        predictions = predict(trainer, dataset)

    pred_labels = np.argmax(predictions, axis=1)
    accuracy = sum(pred_labels == true_labels) / len(true_labels)

    print()
    print("-------------------------------------")
    print("----- MODEL PERFORMANCE METRICS -----")
    print()
    print("accuracy", accuracy)
    print()
    print("confusion matrix")
    print(confusion_matrix(true_labels, pred_labels))
    print()
    print("-------------------------------------")
    print("-------------------------------------")
    print()

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
