"""
profile RAM consumption during inference
"""

import os
from tempfile import mkdtemp
import shutil

import click
from transformers import Trainer, TrainingArguments
from transformers.models.bert.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
# from memory_profiler import profile  # NOTE: need to comment this out to use mprof plotting

import model_utils


@profile
def create_dataset(examples, tokenizer, padding: str = "max_length"):
    return model_utils.Dataset(examples, tokenizer, padding=padding)

@profile
def load_tokenizer(tokenizer_name):
    return model_utils.load_tokenizer(tokenizer_name)

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
    # load text and parse
    text_filepath = "example_data/alice-in-wonderland.txt"
    examples = model_utils.read_text(text_filepath)
    tmp_dir = mkdtemp()

    # load pre-trained model & tokenizer
    tokenizer = load_tokenizer("roberta-base")
    # create dataset
    dataset = create_dataset(examples[:16], tokenizer)

    if model_name.endswith("onnx"):
        session = onnx_load_model(model_name)
        onnx_predict(session, dataset)

    elif model_name.endswith("pt"):
        model = torchscript_load_model(model_name)
        torchscript_predict(model, dataset)

    else:
        # HuggingFace Trainer
        model = load_model(model_name)

        # create trainer
        training_args = TrainingArguments(
            output_dir=tmp_dir,
            do_train=False,
            do_eval=False,
            do_predict=True,
            per_device_eval_batch_size=1
        )
        trainer = Trainer(
            model=model,
            args=training_args
        )

        # run predict
        predictions = predict(trainer, dataset)

    # clean up
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
