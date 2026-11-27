from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "Food Was Good",
    "Food Was Bad",
    "Food Was Not Good"
]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

vocab_size = len(word_index)+1

print("Number of uniqe words :", len(word_index))
print("Padding index : 0")
print("Vocabulary size :", vocab_size)