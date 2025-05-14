import numpy as np
import pandas as pd
from utils.preprocessing import build_preprocess_fn

from models.attentions_tf import (
    build_basic_attention_model,
    build_scaled_dot_attention_model
)
from models.seq2seq_tf import (
    build_simple_rnn_seq2seq_model,
    build_gru_seq2seq_model,
    build_lstm_seq2seq_model
)

def train_model(
    train_in_ids,
    train_dec_tgt,
    val_in_ids,
    val_dec_tgt,
    tokenizer,
    model_type='gru',
    max_input_len=256,
    max_output_len=64,
    embedding_dim=256,
    rnn_units=256,
    num_heads=4,
    ff_dim=256,
    dropout_rate=0.1,
    batch_size=32,
    epochs=15,
):  
    

    # Build model
    if model_type == 'simple_rnn':
        model = build_simple_rnn_seq2seq_model(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            max_input_len=max_input_len,
            max_output_len=max_output_len - 1
        )
    elif model_type == 'gru':
        model = build_gru_seq2seq_model(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            max_input_len=max_input_len,
            max_output_len=max_output_len - 1
        )
    elif model_type == 'lstm':
        model = build_lstm_seq2seq_model(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            max_input_len=max_input_len,
            max_output_len=max_output_len - 1
        )
    elif model_type == 'basic_attention':
        model = build_basic_attention_model(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            max_input_len=max_input_len,
            max_output_len=max_output_len - 1
        )
    elif model_type == 'scaled_attention':
        model = build_scaled_dot_attention_model(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_input_len=max_input_len,
            max_output_len=max_output_len - 1,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train the model with explicit validation data
    history = model.fit(
        train_in_ids ,
        train_dec_tgt,
        validation_data=(val_in_ids, val_dec_tgt),
        batch_size=batch_size,
        epochs=epochs
    )

    return model, history