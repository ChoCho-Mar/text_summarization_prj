
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense

def build_gru_seq2seq_model(vocab_size, embedding_dim, rnn_units, 
                            max_input_len, max_output_len):
    # Encoder
    encoder_inputs = Input(shape=(max_input_len,))
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
    encoder_outputs, state_h = GRU(rnn_units, return_state=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(max_output_len,))
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
    decoder_gru = GRU(rnn_units, return_sequences=True)(decoder_embedding, initial_state=state_h)
    decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_gru)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_lstm_seq2seq_model(vocab_size, embedding_dim, rnn_units, 
                             max_input_len, max_output_len):
    # Encoder
    encoder_inputs = Input(shape=(max_input_len,))
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(rnn_units, return_state=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(max_output_len,))
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(rnn_units, return_sequences=True)(decoder_embedding, initial_state=[state_h, state_c])
    decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_lstm)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def build_simple_rnn_seq2seq_model(vocab_size, embedding_dim, rnn_units, 
                                    max_input_len, max_output_len):
    # Encoder
    encoder_inputs = Input(shape=(max_input_len,))
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
    encoder_outputs, state_h = SimpleRNN(rnn_units, return_state=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(max_output_len,))
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
    decoder_rnn = SimpleRNN(rnn_units, return_sequences=True)(decoder_embedding, initial_state=state_h)
    decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_rnn)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model