"""
Based on the following tutorial:
https://www.tensorflow.org/guide/graphs
"""

import numpy as np
import tensorflow as tf

# Setting the tensor names
c_0 = tf.constant(0, name="c")
print(c_0)

# Already-used names are "uniquified"
c_1 = tf.constant(2, name="c")
print(c_1)

# Name scopes add a prefix to all operations created in the same context
with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c")

    with tf.name_scope("inner"):
        c_3 = tf.constant(3, name="c")

    c_4 = tf.constant(4, name="c")

    with tf.name_scope("inner"):
        c_5 = tf.constant(5, name="c")

print(c_2)
print(c_3)
print(c_4)
print(c_5)

# Operations created outside either context will run on the "best possible"
# device. For example, if you have a GPU and a CPU available, and the operation
# has a GPU implementation, TensorFlow will choose the GPU
weights = tf.random_normal((5, 1))

# Operations created in this context will be pinned to the first CPU
with tf.device("/device:CPU:0"):
    weights2 = tf.random_normal((5, 1))

# It's better to start a session using the context manager
with tf.Session() as sess:
    sess.run(weights2)

# Let's create some tensor, variables and operation
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
    # Run the initializer on w
    sess.run(init_op)

    # Evaluate output
    print(sess.run(output))

    # Evaluate y and output. Note that y will only be computed once
    y_val, output_val = sess.run([y, output])
    print(y_val)
    print(output_val)

# Define a placeholder and a computation that depends on it
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
    # Feeding a value changes the result that is returned when you evaluate y
    print(sess.run(y, {x: [1.0, 2.0, 3.0]}))
    print(sess.run(y, {x: [0.0, 0.0, 5.0]}))

    # This raises an exception because the placeholder is not feed
    # sess.run(y)

    # This raises an exception because the shapes do not match
    # sess.run(y, {x: 37.0})

# We can pass options to the session run and collect execution metadata
y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
    # Define options for the `sess.run()` call.
    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.trace_level = tf.RunOptions.FULL_TRACE

    # Define a container for the returned metadata.
    metadata = tf.RunMetadata()

    sess.run(y, options=options, run_metadata=metadata)

    # Print the subgraphs that executed on each device
    print(metadata.partition_graphs)

    # Print the timings of each operation that executed
    print(metadata.step_stats)

# We can create graphs to collect tensorflow operations
g_1 = tf.Graph()

with g_1.as_default():
    # Operations created in this scope will be added to g_1
    c = tf.constant("Node in g_1")

    # Sessions created in this scope will run operations from g_1
    sess_1 = tf.Session()

g_2 = tf.Graph()

with g_2.as_default():
    # Operations created in this scope will be added to g_2
    d = tf.constant("Node in g_2")

# We can set the graph to use in a session
sess_2 = tf.Session(graph=g_2)

# We can test that everything is correct
assert c.graph is g_1
assert sess_1.graph is g_1
assert d.graph is g_2
assert sess_2.graph is g_2

# Print all the operations in the default graph
g = tf.get_default_graph()
print(g.get_operations())

