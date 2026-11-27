sentences = [
    "Food Was Good",
    "Food Was Bad",
    "Food Was Not Good"
]

labels = [1,0,0]

for sentence, label in zip(sentences, labels):
    sentiment = "positive" if label == 1 else "Negative"

    print("Sentence :", sentence)
    print("Label :", label)
    print("Meaning :", sentiment)
    print("--------------------------")

