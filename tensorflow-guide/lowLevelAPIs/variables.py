"""
Based on the following tutorial:
https://www.tensorflow.org/guide/variables
"""

import numpy as np
import tensorflow as tf

# Create a session
sess = tf.Session()

# Create some variables
my_variable = tf.get_variable("my_variable", [1, 2, 3])
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)
other_variable = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant([23, 42]))

# There are to ways to set a variable as not trainable
my_local = tf.get_variable("my_local", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
my_non_trainable = tf.get_variable("my_non_trainable", shape=(), trainable=False)

# Add a variable to a used defined collection
tf.add_to_collection("my_collection_name", my_local)

# Print the variables in different collections
print(tf.get_collection("my_collection_name"))
print(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

# Placing a variable in a given device
with tf.device("/device:CPU:0"):
    v = tf.get_variable("v", [1])

# Initialize all global variables
sess.run(tf.global_variables_initializer())

# One can also initialize variables individually
sess.run(my_variable.initializer)

# Find the variables that have not been initialized
print(sess.run(tf.report_uninitialized_variables()))

# Correct way to initialize a variable that depends on another variable
x = tf.get_variable("x", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=x.initialized_value() + 1)

# We can now use the variable as a normal tensor
l = w + 1

# We can assign values to already defined variables
assignment = x.assign_add(1)
sess.run(tf.global_variables_initializer())
sess.run(x)
sess.run(assignment)
sess.run(x)

# And we can read the assigned value
sess.run(tf.global_variables_initializer())
print(sess.run(x.read_value()))

with tf.control_dependencies([assignment]):
    value = x.read_value()

print(sess.run(value))


# Lets create a function that defines some variables
def conv_relu(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding="SAME")

    return tf.nn.relu(conv + biases)


# Using the function twice will make it crash because the variables are defined twice
input1 = tf.random_normal([1, 10, 10, 32])
input2 = tf.random_normal([1, 20, 20, 32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
# x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape=[32])


# We can avoid that using variable scopes
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        x = conv_relu(input_images, [5, 5, 32, 32], [32])

    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(x, [5, 5, 32, 32], [32])


with tf.variable_scope("model"):
    output1 = my_image_filter(input1)

with tf.variable_scope("model", reuse=True):
    output2 = my_image_filter(input2)

# Or another way
with tf.variable_scope("new_model") as scope:
    output1 = my_image_filter(input1)
    scope.reuse_variables()
    output2 = my_image_filter(input2)

