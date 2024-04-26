import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np
import os
import re
import keras
import mlflow
import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Positional Encoding
def get_positional_encoding(seq_length, d_model):
    positions = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = positions * angle_rates
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return pos_encoding


def transformer_block(x, num_heads, d_model, dff, rate, training):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(rate)(attn_output, training=training)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output, training=training)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


# Build the Transformer Model for Text Summarization
def build_model(max_len_input, max_len_output, vocab_size, num_heads=8, d_model=128, dff=512, rate=0.1):
    # Input to the encoder
    encoder_inputs = Input(shape=(max_len_input,), name="encoder_input")
    encoder_embedding = Embedding(vocab_size, d_model, name="encoder_embedding")(encoder_inputs)
    encoder_pos_encoding = get_positional_encoding(max_len_input, d_model)
    encoder_embedding += encoder_pos_encoding

    # Encoder
    encoder_output = encoder_embedding
    for _ in range(4):
        encoder_output = transformer_block(encoder_output, num_heads, d_model, dff, rate, training=True)

    # Input to the decoder
    decoder_inputs = Input(shape=(max_len_output,), name="decoder_input")
    decoder_embedding = Embedding(vocab_size, d_model, name="decoder_embedding")(decoder_inputs)
    decoder_pos_encoding = get_positional_encoding(max_len_output, d_model)
    decoder_embedding += decoder_pos_encoding

    # Decoder
    decoder_output = decoder_embedding
    for _ in range(4):
        decoder_output = transformer_block(decoder_output, num_heads, d_model, dff, rate, training=True)

    # Output layer
    outputs = Dense(vocab_size, activation="softmax")(decoder_output)


    # Define the model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

data = np.load('../data/dataset_cleaned.npz', allow_pickle=True)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
X_val = data['X_val']
y_val = data['y_val']

X = np.concatenate((X_train, X_test, X_val))
y = np.concatenate((y_train, y_test, y_val))

max_len_input = len(max(X, key=len).split())
max_len_output = len(max(y, key=len).split())
print(max_len_input)

tokenizer = Tokenizer(num_words=max_len_input)
tokenizer.fit_on_texts(X)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)
y_train = tokenizer.texts_to_sequences(y_train)
y_test = tokenizer.texts_to_sequences(y_test)
y_val = tokenizer.texts_to_sequences(y_val)

vocab_size = len(tokenizer.get_config()['word_counts'])
print("VOCAB SIZE: " + str(vocab_size))

# Standardize Data by padding sequences
X_train = pad_sequences(X_train, maxlen=max_len_input, padding ='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_len_input, padding ='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=max_len_input, padding ='post', truncating='post')
y_train = pad_sequences(X_train, maxlen=max_len_output, padding ='post', truncating='post')
y_test = pad_sequences(X_test, maxlen=max_len_output, padding ='post', truncating='post')
y_val = pad_sequences(X_val, maxlen=max_len_output, padding ='post', truncating='post')

create_new_model = True

version = 0

models = os.listdir("../models/transformer/")
sub1 = "transformer_model_v"
sub2 = ".keras"
s=str(re.escape(sub1))
e=str(re.escape(sub2))

print(models)

for m in models:
    v = re.findall(s+"(.*)"+e,m)[0]
    if int(v) > version:
        version = int(v)

print("VERSION: " + str(version))
                    
if create_new_model:
    
    
    version = version + 1

    name = "transformer_model_v" + str(version) + ".keras" 

    mlflow.set_experiment(name)
    experiment = mlflow.get_experiment_by_name(name)
    
    mlflow.autolog()
    
    with mlflow.start_run():
        
        model = build_model(max_len_input, max_len_output, vocab_size)

        log_dir = "../logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = ModelCheckpoint("../models/transformer/" + name, save_best_only=True)
        early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
        reduce_lr_callback = ReduceLROnPlateau(patience=5, factor=0.1)

        history = model.fit(
            [X_train, y_train], y_train, batch_size=32, epochs=4, verbose=True, validation_data=([X_val, y_val], y_val), callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback, reduce_lr_callback]
        )

        model.save("../models/transformer/transformer_model_v" + str(version) + ".keras")
else:
    model = keras.models.load_model("../models/transformer/transformer_model_v" + str(version) + ".keras")

model.summary()

def predict_summary(input_text, tokenizer, model, max_len_input, max_len_output):
    # Tokenize and pad the input text
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq_padded = pad_sequences(input_seq, maxlen=max_len_input, padding='post', truncating='post')

    # Start the decoder sequence with the start token (assuming `1` is the start token index)
    decoder_input = [1]  # Start token
    decoder_input_padded = pad_sequences([decoder_input], maxlen=max_len_output, padding='post', truncating='post')

    # Initialize the summary with empty list
    summary = []

    for _ in range(max_len_output - 1):  # We loop up to max_len_output - 1 steps
        predictions = model.predict([input_seq_padded, decoder_input_padded])
        next_word_id = np.argmax(predictions[0, len(decoder_input) - 1, :])

        # Append to summary
        summary.append(next_word_id)

        # Print each predicted word
        predicted_word = tokenizer.index_word.get(next_word_id, '?')
        print("Predicted word:", predicted_word)

        # Add the predicted word to the sequence
        decoder_input.append(next_word_id)
        decoder_input_padded = pad_sequences([decoder_input], maxlen=max_len_output, padding='post', truncating='post')

        if next_word_id == 2:  # Assuming `2` is the end token index
            break

    predicted_sequence = tokenizer.sequences_to_texts([summary])
    return predicted_sequence



# Example usage:
text_to_summarize = """KerasNLP is a natural language processing library that works natively with TensorFlow, JAX, or PyTorch. Built on Keras 3, these models, layers, metrics, and tokenizers can be trained and serialized in any framework and re-used in another without costly migrations.

KerasNLP supports users through their entire development cycle. Our workflows are built from modular components that have state-of-the-art preset weights when used out-of-the-box and are easily customizable when more control is needed.

This library is an extension of the core Keras API; all high-level modules are Layers or Models that receive that same level of polish as core Keras. If you are familiar with Keras, congratulations! You already understand most of KerasNLP."""
summary = predict_summary(text_to_summarize, tokenizer, model, max_len_input=100, max_len_output=50)
print("Generated Summary:", summary)

