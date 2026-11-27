from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "Food Was Good",
    "Food Was Bad",
    "Food Was Not Good"
]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

print("Original Sequences.")
for sequence in sequences:
    print(sequence, "Length :", len(sequence))

print("All sequences of different length")

max_length = 4

padded_sequences = pad_sequences(
    sequences,
    maxlen = max_length, 
    padding = "pre"
)

for sentence, sequence, padded in zip(sentences, sequences, padded_sequences):
    print("Sentence :", sentence)
    print("Original sequence :", padded)
    print("--------------------------------------")
