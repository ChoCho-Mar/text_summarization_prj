import pandas as pd
import numpy as np

def build_preprocess_fn(tokenizer, max_input_len=512, max_target_len=128):
    def preprocess(sample):
        # Tokenize input (dialogue)
        inputs = tokenizer(
            sample['dialogue'],
            truncation=True,
            max_length=max_input_len,
            padding='max_length',
            return_tensors='np'
        )

        # Tokenize output (summary)
        targets = tokenizer(
            sample['summary'],
            truncation=True,
            max_length=max_target_len,
            padding='max_length',
            return_tensors='np'
        )

        decoder_input_ids = targets['input_ids'][:, :-1]  # all tokens except the last
        decoder_target_ids = targets['input_ids'][:, 1:]  # all tokens except the first

        return {
            'encoder_input_ids': inputs['input_ids'],
            'decoder_input_ids': decoder_input_ids,
            'decoder_target_ids': decoder_target_ids
        }
    return preprocess

def load_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['dialogue', 'summary'])
    return [{'dialogue': row['dialogue'], 'summary': row['summary']} for _, row in df.iterrows()]

def preprocess_dataset(dataset, preprocess_fn):
    processed = [preprocess_fn(sample) for sample in dataset]
    encoder_input_ids = np.vstack([item['encoder_input_ids'] for item in processed])
    decoder_input_ids = np.vstack([item['decoder_input_ids'] for item in processed])
    decoder_target_ids = np.vstack([item['decoder_target_ids'] for item in processed])
    return [encoder_input_ids, decoder_input_ids], decoder_target_ids
