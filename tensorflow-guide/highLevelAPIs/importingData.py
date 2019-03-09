"""
Based on the following tutorial:
https://www.tensorflow.org/guide/datasets
"""

import numpy as np
import tensorflow as tf

# Create a session
sess = tf.Session()

# Create a new dataset
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))

# Get the elements type and shape
print(dataset1.output_types)
print(dataset1.output_shapes)

# Another example
dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)
print(dataset2.output_shapes)

# And another that uses the previous datasets
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)
print(dataset3.output_shapes)

# One can name the element components
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)
print(dataset.output_shapes)

# We can apply operations on the data set elements
dataset1 = dataset1.map(lambda x: x * x)
dataset3 = dataset3.filter(lambda x, y: x[0] > 0.5)

# The most simple iterator is the one shot iterator
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
    value = sess.run(next_element)
    assert i == value

# One can also create a initialize iterator
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer, feed_dict={max_value: 10})

for i in range(10):
    value = sess.run(next_element)
    assert i == value

sess.run(iterator.initializer, feed_dict={max_value: 100})

for i in range(100):
    value = sess.run(next_element)
    assert i == value

# Define a training and validation datasets with the same structure
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# Create a reinitializable iterator for the training and the validation datasets
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()
training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in with the iterators are reused
for _ in range(20):
    sess.run(training_init_op)
    
    for _ in range(100):
        sess.run(next_element)
    
    sess.run(validation_init_op)
    
    for _ in range(50):
        sess.run(next_element)

# Define a training and validation datasets
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# One can use feedable iterators with a variety of different kinds of iterator
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop 5 times, alternating between training and validation
counter = 0

while counter < 5:
    for _ in range(200):
        sess.run(next_element, feed_dict={handle: training_handle})
    
    sess.run(validation_iterator.initializer)
    
    for _ in range(50):
        sess.run(next_element, feed_dict={handle: validation_handle})
    
    counter += 1

# Iterators raise an exception when they reach the end of the iteration
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
result = tf.add(next_element, next_element)
sess.run(iterator.initializer)

while True:
    try:
        sess.run(result)
    except tf.errors.OutOfRangeError:
        break

# One can create batches from a dataset
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))
print(sess.run(next_element))
print(sess.run(next_element))

