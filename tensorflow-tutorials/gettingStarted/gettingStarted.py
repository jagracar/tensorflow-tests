'''
 Based on the following tutorial:
   https://www.tensorflow.org/get_started/get_started
'''

import tensorflow as tf


# Create some nodes
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(node1, node2)

# Create a session
sess = tf.Session()

# Create an additive graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# Run the computational graph
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# Create a more coplicated graph and run it
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))

# Create some variables
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Initialize the variables
init = tf.global_variables_initializer()
sess.run(init)

# Create a linear model graph and run it for several values
x = tf.placeholder(tf.float32)
linear_model = W * x + b
x_train = [1, 2, 3, 4]
print(sess.run(linear_model, {x:x_train}))

# Create a loss graph
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Calculate the loss value for the variable defaults
y_train = [0, -1, -2, -3]
print(sess.run(loss, {x:x_train, y:y_train}))

# Reassign the variable values
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

# Calculate the new loss value
print(sess.run(loss, {x:x_train, y:y_train}))

# Create an optimizer and use it to minimize the loss
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Start with the initial variable values
sess.run(init) 

# Minimize the loss in 1000 iterations
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# Print the final variable values
print(sess.run([W, b]))

# Evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
