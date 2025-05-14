import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, GRU, Concatenate, Attention
from tensorflow.keras.layers import MultiHeadAttention, Add

def build_basic_attention_model(vocab_size, embedding_dim, rnn_units, max_input_len, max_output_len):
    # Encoder
    encoder_inputs = Input(shape=(max_input_len,))
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
    encoder_outputs, encoder_state = GRU(rnn_units, return_sequences=True, return_state=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(max_output_len,))
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
    decoder_gru_outputs, _ = GRU(rnn_units, return_sequences=True, return_state=True)(decoder_embedding, initial_state=encoder_state)

    # Attention
    attention_layer = Attention()  # This is Bahdanau-style attention
    context_vector = attention_layer([decoder_gru_outputs, encoder_outputs])

    # Concatenate context and decoder output
    concat = Concatenate()([context_vector, decoder_gru_outputs])

    # Final prediction layer
    output = Dense(vocab_size, activation='softmax')(concat)

    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_scaled_dot_attention_model(vocab_size, embedding_dim, num_heads, ff_dim, max_input_len, max_output_len, dropout_rate=0.1):
    # Encoder
    encoder_inputs = Input(shape=(max_input_len,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)

    # Encoder self-attention block
    encoder_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(encoder_embedding, encoder_embedding)
    encoder_attention = Dropout(dropout_rate)(encoder_attention)
    encoder_attention = Add()([encoder_embedding, encoder_attention])
    encoder_attention = LayerNormalization()(encoder_attention)

    # Feed-forward layer
    ff_1 = Dense(ff_dim, activation='relu')(encoder_attention)
    ff_2 = Dense(embedding_dim)(ff_1)
    encoder_outputs = LayerNormalization()(Add()([encoder_attention, ff_2]))

    # Decoder
    decoder_inputs = Input(shape=(max_output_len,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)

    # Decoder masked self-attention
    causal_mask = tf.linalg.band_part(tf.ones((max_output_len, max_output_len)), -1, 0)
    decoder_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(
        decoder_embedding, decoder_embedding, attention_mask=causal_mask
    )
    decoder_self_attention = Dropout(dropout_rate)(decoder_self_attention)
    decoder_self_attention = Add()([decoder_embedding, decoder_self_attention])
    decoder_self_attention = LayerNormalization()(decoder_self_attention)

    # Decoder cross-attention
    decoder_cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(
        decoder_self_attention, encoder_outputs
    )
    decoder_cross_attention = Dropout(dropout_rate)(decoder_cross_attention)
    decoder_cross_attention = Add()([decoder_self_attention, decoder_cross_attention])
    decoder_cross_attention = LayerNormalization()(decoder_cross_attention)

    # Feed-forward layer
    ff_1 = Dense(ff_dim, activation='relu')(decoder_cross_attention)
    ff_2 = Dense(embedding_dim)(ff_1)
    decoder_output = LayerNormalization()(Add()([decoder_cross_attention, ff_2]))

    # Final output
    final_output = Dense(vocab_size, activation='softmax')(decoder_output)

    model = Model([encoder_inputs, decoder_inputs], final_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model