"""
Based on the following tutorial:
https://www.tensorflow.org/guide/low_level_intro
"""

import numpy as np
import tensorflow as tf

# Create some constant tensors
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a + b
print(a)
print(b)
print(total)

# Create a session
sess = tf.Session()

# Evaluate the tensors
print(sess.run(total))
print(sess.run({"a and b" : (a, b), "total" : total}))

# Create an additive graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
z = a + b

# Run the computational graph
print(sess.run(z, feed_dict={a : 3, b : 4.5}))
print(sess.run(z, feed_dict={a : [1, 3], b : [2, 4]}))

# Create a data iterator
my_data = [[0, 1], [2, 3], [4, 5], [6, 7]]
dataset = tf.data.Dataset.from_tensor_slices(my_data)
next_item = dataset.make_one_shot_iterator().get_next()

# Run the data iterator
while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

# Another possible iterator
random_data = tf.random_normal([10, 3])
dataset = tf.data.Dataset.from_tensor_slices(random_data)
iterator = dataset.make_initializable_iterator()
next_item = iterator.get_next()

sess.run(iterator.initializer)

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

# Create a dense layer: outputs = activation(inputs * kernel + bias)
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1, use_bias=True)
y = linear_model(x)

# Initialize the variables
init = tf.global_variables_initializer()
sess.run(init)

# Execute the layer
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
print(sess.run(linear_model.kernel))
print(sess.run(linear_model.bias))

# Create a features input layer
features = {
    "sales" : [[5], [10], [8], [9]],
    "department" : ["sports", "sports", "gardening", "gardening"]
}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="department", vocabulary_list=["sports", "gardening"])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column("sales"),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

# Initialize the inputs
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess.run((var_init, table_init))

# Print the inputs
print(sess.run(inputs))

# Lets train a regression model. First define the inputs and true outputs
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

# Define a simple linear model
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

# Evaluate the predictions
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))

# Define the loss function
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

# Define the variable optimizer and define the optimize method
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Fit the model iteratively
for i in range(1000):
    _, loss_value = sess.run((train, loss))
    if i % 100 == 0:
        print("iteration %s, loss: %s" % (i, loss_value))

# Print the predicted values
print(sess.run(y_pred))

# Print the model variables
print("slope: ", sess.run(linear_model.kernel))
print("intersect: ", sess.run(linear_model.bias))
