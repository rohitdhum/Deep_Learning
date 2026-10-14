import tensorflow as tf

tensor1 = tf.constant([10,20,30])
tensor2 = tf.constant([1,2,3])

Addition = tf.add(tensor1, tensor2)
print("Addition is :", Addition)               # 11,22,33

Substraction = tf.subtract(tensor1, tensor2)
print("Substraction is :", Substraction)       # 9,18,27

Multiplication = tf.multiply(tensor1, tensor2)
print("Multiplication is :", Multiplication)   # 10,40,90

Division = tf.divide(tensor1, tensor2)
print("Division is :", Division)               # 10.0, 10.0, 10.0

Square = tf.square(tensor1)
print("Square is :", Square)                   # 100,400,900

Sum = tf.reduce_sum(tensor1)
print("Reduce Sum :", Sum)                     # 60

Mean = tf.reduce_mean(tensor1)
print("Mean is :", Mean)                       # 20