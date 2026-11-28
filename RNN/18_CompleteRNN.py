import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Step 1 : Load the Data

train_sentences = [
    "Food Was Good",
    "Food Was Bad",
    "Food Was Excellent",
    "Food Was Terriable",
    "Service Was Good",
    "Service Was Bad",
    "Service Was Excellent",
    "Service Was Terriable",
    "Ambience Was Good",
    "Ambience Was Bad",
    "Ambience Was Excellent",
    "Ambience Was Terriable"
]

train_labels = [
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
]

# Step 2 : Tokenization

tokenizer = Tokenizer(oov_token = "<OOV>")

tokenizer.fit_on_texts(train_sentences)

# Step 3 : Convert training data into sequence

train_sequence = tokenizer.texts_to_sequences(train_sentences)

print("Training sequences :")

for sentence, sequence in zip(train_sentences, train_sequence):
    print(sentence, "->", sequence)