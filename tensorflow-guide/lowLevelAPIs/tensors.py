"""
Based on the following tutorial:
https://www.tensorflow.org/guide/tensors
"""

import numpy as np
import tensorflow as tf

# Create a session
sess = tf.Session()

# Some rank 0 tensors
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

# Some rank 1 tensors
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

# Some rank 2 tensors
mymat = tf.Variable([[7], [11]], tf.int16)
myxor = tf.Variable([[False, True], [True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
mymatC = tf.Variable([[7], [11]], tf.int32)

# A rank 4 tensor
my_image = tf.zeros([10, 299, 299, 3])

# Initialize the variables
init = tf.global_variables_initializer()
sess.run(init)

# Print the rank of a tensor
print(sess.run(tf.rank(my_image)))

# Slicing tensors
slice = myxor[1]
print(sess.run(slice))

# Changing the shape of tensors
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])
matrixB = tf.reshape(matrix, [3, -1])
matrixAlt = tf.reshape(matrixB, [4, 3, -1])

# Cast a constant integer tensor into a floating point tensor
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
print(float_tensor.dtype)

# Using the eval method to evaluate a tensor
constant = tf.constant([1, 2, 3])
tensor = constant * constant
print(tensor.eval(session=sess))

# Same with a placeholder tensor
p = tf.placeholder(tf.float32)
t = p + 1.0
t.eval(feed_dict={p:2.0}, session=sess)
