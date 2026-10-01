import numpy as np

# Step 1 : Define Input features ie X
#                 [X1, X2, X3]
input = np.array([2.0,3.0,4.0])
print("X :",input)

# Step 2 : Define Weights ie W
#                  [w1, w2, w3]                 
weights = np.array([0.5,0.3,0.2])
print("W :",weights)

# Step 3 : Define Bias ie b
#       b
bias = 1.0
print("b :",bias)

# Step 4 : Calculate weighted sum ie Z
# Z = x1w1 + x2w2 + x3w3 + b
# Z = (2.0*0.5) + (3.0*0.3) + (4.0*0.2) + 1.0

z = np.dot(input,weights) + bias
print("Z :",z)

# Step 5 : Activation function (ReLU)
def ReLU(x):
    return max(0,x)

# Step 6 : Final Output
Y = ReLU(z)
print("Y :", Y)


