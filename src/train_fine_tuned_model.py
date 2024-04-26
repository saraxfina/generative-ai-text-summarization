import pandas as pd
import tensorflow as tf
import numpy as np
import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 

from transformers import AutoTokenizer, TFBartForConditionalGeneration
import pandas as pd


data = np.load('../data/dataset_cleaned.npz', allow_pickle=True)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
X_val = data['X_val']
y_val = data['y_val']

max_len_input = len(max(X_train, key=len).split())
max_len_output = len(max(y_train, key=len).split())

train_source_texts = X_train.tolist()
train_target_texts = y_train.tolist()
val_source_texts = X_val.tolist()
val_target_texts = y_val.tolist()

def prepare_data_for_bart(source_texts, target_texts, max_len=512):
    """
    Prepares data for BART model training.
    Args:
      source_texts: List of strings containing document text.
      target_texts: List of strings containing corresponding summaries.
      max_len: Maximum sequence length for tokenization (optional).
    Returns:
      A dictionary containing tokenized inputs and labels.
    """
    inputs = tokenizer(source_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="tf")
    labels = tokenizer(target_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="tf")

    print("Input IDs shape:", inputs['input_ids'].shape)
    print("Labels IDs shape:", labels['input_ids'].shape)

    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "decoder_attention_mask": labels.attention_mask,
        "labels": labels.input_ids  # labels are shifted for teacher forcing
    }

def convert_to_dataset(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda x: {key: tf.reshape(val, [-1]) for key, val in x.items()})
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Define model name
model_name = "facebook/bart-base"

# Load tokenizer and pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFBartForConditionalGeneration.from_pretrained(model_name)

# Load model directly 
#tokenizer = AutoTokenizer.from_pretrained("google/mt5-small") 
#model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# Prepare and convert training and validation data
train_data_dict = prepare_data_for_bart(train_source_texts, train_target_texts)
val_data_dict = prepare_data_for_bart(val_source_texts, val_target_texts)
train_dataset = convert_to_dataset(train_data_dict, batch_size=8)
val_dataset = convert_to_dataset(val_data_dict, batch_size=8)

import os 
import re
import tf_keras

# Define optimizer and loss function
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

create_new_fine_tuned_model = True

version = 0
models = os.listdir("../models/fine_tuned/")
sub1 = "fine_tuned_model_v"
sub2 = ".keras"
s=str(re.escape(sub1))
e=str(re.escape(sub2))

for m in models:
    v = re.findall(s+"(.*)"+e,m)[0]
    if int(v) > version:
        version = int(v)

                    
if create_new_fine_tuned_model:
    
    
    version = version + 1

    name = "fine_tuned_model_v" + str(version) + ".keras" 

    mlflow.set_experiment(name)
    experiment = mlflow.get_experiment_by_name(name)
    
    mlflow.autolog()
    
    with mlflow.start_run():

        # Training loop
        epochs = 1

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            batch_count = 0
            for batch in train_dataset:
                batch_count += 1
                print(f"Training batch {batch_count}...")
                with tf.GradientTape() as tape:
                    outputs = model(**batch)
                    loss_value = loss(batch["labels"], outputs.logits)
                    print(f"Batch {batch_count} loss: {loss_value.numpy()}")
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                tf.keras.backend.clear_session()  # Clear memory

            print(f"Completed {batch_count} batches.")
            
            tokenizer.save_pretrained("../models/fine_tuned/fine_tuned_model_v" + str(version) + ".keras")
            model.save_pretrained("../models/fine_tuned/fine_tuned_model_v" + str(version) + ".keras")

            # Evaluate on validation data
            # val_loss, val_acc = model.evaluate(val_dataset)
            # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


else:
    model = keras.models.load_model("../models/fine_tuned/fine_tuned_model_v" + str(version) + ".keras")
    

model.summary()


