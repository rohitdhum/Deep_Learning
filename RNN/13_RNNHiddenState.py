words = ["Food", "Was", "Not", "Good"]

hidden_state = "empty memory"

print("Input Tokens :", words)

print("Initial Hidden State :", hidden_state)

for index, word in enumerate(words):
    print("Timestep :", index+1)
    print("Current word :", word)
    print("Previous memory :", hidden_state)

    hidden_state = "memory after reading" + " ".join(words[:index + 1]) + ""
    print("Updated Memory :", hidden_state)
    
    print("-"*30)
