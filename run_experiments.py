import numpy as np
import pandas as pd
import os
import argparse
import logging
import yaml
import evaluate
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer

from utils.preprocessing import (
    build_preprocess_fn,
    load_dataset_from_csv,
    preprocess_dataset
)
from training.train_model import train_model

logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def decode_predictions(pred_ids, tokenizer):
    # Convert predicted token IDs back to text
    texts = []
    for ids in pred_ids:
        # Stop at first [PAD] or [EOS] token (id 0 or 1 for T5 tokenizer)
        end_idx = np.where((ids == tokenizer.pad_token_id) | (ids == tokenizer.eos_token_id))[0]
        cutoff = end_idx[0] if len(end_idx) > 0 else len(ids)
        text = tokenizer.decode(ids[:cutoff], skip_special_tokens=True)
        texts.append(text)
    return texts

def train_models(model_types, train_inputs, train_targets, val_inputs, val_targets, tokenizer, epochs, save_path):
    histories = {}

    for mtype in model_types:
        print("training model : ",mtype)
        model, history = train_model(
            train_inputs,
            train_targets,
            val_inputs,
            val_targets,
            tokenizer,
            model_type=mtype,
            epochs = epochs)
        
        # Save model
        model_file = os.path.join(save_path, f"{mtype}_model.h5")
        model.save(model_file)
        logging.info(f"Model saved: {model_file}")
        histories[mtype] = history.history['val_accuracy'][-1]

    return histories

def evaluate_models(model_types, test_inputs, test_targets, save_path):
    for mtype in model_types:
        model_file = os.path.join(save_path, f"{mtype}_model.h5")
        if not os.path.exists(model_file):
            logging.warning(f"Model not found: {model_file}")
            continue
        model = load_model(model_file, compile=True)
        results = model.evaluate(test_inputs, test_targets, verbose=1)
        logging.info(f"{mtype} : Test Loss and Metrics: {results}")


def display_predictions(model, test_inputs, test_data, tokenizer):

    preds = model.predict(test_inputs, verbose=0)
    pred_ids = np.argmax(preds, axis=-1)
    generated_summaries = decode_predictions(pred_ids, tokenizer)
    true_summaries = [sample['summary'] for sample in test_data]

    return generated_summaries, true_summaries

def calculate_rouge_scores(generated_summaries, true_summaries):
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=generated_summaries, references=true_summaries)
    return rouge_results

def main(config_path):
    config = load_config(config_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_name"])

    # Load data
    train_data = load_dataset_from_csv(config["data"]["train_path"])
    val_data = load_dataset_from_csv(config["data"]["val_path"])
    test_data = load_dataset_from_csv(config["data"]["test_path"])

    num_samples = config["prediction"]["num_samples"]

    # Preprocess
    preprocess_fn = build_preprocess_fn(
        tokenizer,
        config["training"]["max_input_len"],
        config["training"]["max_output_len"]
    )
    train_inputs, train_targets = preprocess_dataset(train_data, preprocess_fn)
    val_inputs, val_targets = preprocess_dataset(val_data, preprocess_fn)
    test_inputs, test_targets = preprocess_dataset(test_data, preprocess_fn)

    # Train all models
    model_types = config["model"]["model_types"]
    save_path = config["output"]["save_model_path"]
    os.makedirs(save_path, exist_ok=True)

    histories = train_models(
        model_types,
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        tokenizer,
        config["training"]["epochs"],
        save_path
    )

    # Validation Accuracy
    print("\nValidation Accuracy Comparison:")
    for model_type, val_acc in histories.items():
        print(f"{model_type}: {val_acc:.4f}")

    # Evaluate test dataset
    evaluate_models(model_types, test_inputs, test_targets, save_path)

    # Predictions (for the last model trained)

    # Split the dataset as the Memory is not enough
    data_split = config["prediction"]["data_split"] # first 400 samples will be predicted

    test_data = test_data[:data_split]
    inputs = [sample['dialogue'] for sample in test_data]

    test_encoder_in, test_decoder_in = test_inputs
    test_encoder_in = test_encoder_in[:data_split]
    test_decoder_in = test_decoder_in[:data_split]

    for model_type in model_types:
        print(model_type," : predictions")
        model_path = os.path.join(save_path, f"{model_type}_model.h5")
        if os.path.exists(model_path):
            model = load_model(model_path, compile=True)
            generated_summaries, true_summaries= display_predictions(model, [test_encoder_in, test_decoder_in], test_data, tokenizer)
            
            for i in range(num_samples):
                print(f"\nSample {i + 1}")
                print("Input Dialogue:\n", inputs[i])
                print("Generated Summary:\n", generated_summaries[i])
                print("True Summary:\n", true_summaries[i])
            
            rouge_results = calculate_rouge_scores(generated_summaries, true_summaries)
            for key in rouge_results:
                print(f"{key} : {rouge_results[key]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    main(args.config)