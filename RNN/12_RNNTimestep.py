sentence = "Food Was Not Good"
# Timestep    1   2   3   4 
# Token       1   2   5   3
# Embedding  [0.7,0.9] [0.7,0.9] [0.7,0.9] [0.7,0.9] [0.7,0.9]
words = sentence.split()

print("Actual Sentence is :", sentence)

for index, word in enumerate(words):
    print("Timestep :", index+1, ":", word)

