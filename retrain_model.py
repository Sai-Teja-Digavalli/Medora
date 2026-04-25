import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=[2, 0]) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=[2, 0]), axis=1)
        context_vector = tf.reduce_sum(attention_weights[..., tf.newaxis] * x, axis=1)
        return context_vector

# Load and preprocess the dataset
file_path = 'Symptom2Disease.csv'
data = pd.read_csv(file_path)
data = data[['label', 'text']]

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data['text'] = data['text'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])

# Prepare data for feature extraction and modeling
X = data['text']
y = data['label_encoded']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenize text for LSTM
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_sequence_length = 150
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Build enhanced LSTM model with Attention mechanism
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length)(input_layer)
lstm_layer = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(embedding_layer)
attention_layer = AttentionLayer()(lstm_layer)
dense_layer = Dense(128, activation='relu')(attention_layer)
dropout_layer = Dropout(0.4)(dense_layer)
output_layer = Dense(len(label_encoder.classes_), activation='softmax')(dropout_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Custom Callback to Display Precision, Recall, and F1-Score After Every Epoch
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred_probs = model.predict(X_test_pad, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\nEpoch {epoch+1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Train the model with the custom callback
history = model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), 
                    epochs=25, batch_size=32, verbose=1, callbacks=[MetricsCallback()])

# Evaluate the model
accuracy = model.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"Enhanced Model with Attention Accuracy: {accuracy * 100:.2f}%")

# Function to predict disease from user input
def predict_disease(symptoms):
    symptoms_cleaned = clean_text(symptoms)
    seq = tokenizer.texts_to_sequences([symptoms_cleaned])
    pad = pad_sequences(seq, maxlen=max_sequence_length)
    pred = model.predict(pad)
    predicted_label = label_encoder.inverse_transform([np.argmax(pred)])[0]
    return predicted_label


model.save("disease_prediction_model.keras")

import pickle

# Assuming tokenizer and label_encoder are already created in your training script
with open("preprocessing.pkl", "wb") as f:
    pickle.dump({"tokenizer": tokenizer, "label_encoder": label_encoder}, f)

print("preprocessing.pkl saved successfully!")


# Example user interaction removed to run non-interactively
# user_input = input("Enter symptoms: ")
# prediction = predict_disease(user_input)
# print(f"Predicted Disease: {prediction}")
