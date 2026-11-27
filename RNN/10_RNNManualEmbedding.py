Embedding = {
    0 : [0.0,0.0,0.0],  # Padding 
    1 : [0.2,0.4,0.1],  # Food
    2 : [0.3,0.1,0.5],  # Was
    3 : [0.8,0.7,0.9],  # Good
    4 : [0.1,0.2,0.9],  # Bad
    5 : [0.9,0.3,0.2]   # Not
}

sequence = [1,2,5,3]

print("Sequence for 'Food Was Not Good' is :", sequence)

for token in sequence:
    print("Token :", token)
    print("Vector :", Embedding[token])
    print("-----------------------------------")
    