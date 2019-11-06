"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from utils import plotUtils
from utils import statsUtils

# Load the IMDB data set restricting it to 10000 words
num_words = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

# Multi-hot-encode the train and test data sets
train_data = statsUtils.create_multi_hot_encoding(train_data, num_words)
test_data = statsUtils.create_multi_hot_encoding(test_data, num_words)

# Define the baseline model
baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=[num_words]),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# Compile the model
baseline_model.compile(optimizer="adam",
                       loss="binary_crossentropy",
                       metrics=["accuracy", "binary_crossentropy"])

# Print the model summary
baseline_model.summary()

# Fit the model
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

# Plot the training history
plotUtils.plot_training_history(baseline_history)

# Create a smaller model
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=[num_words]),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy", "binary_crossentropy"])

smaller_model.summary()

# Fit the smaller model
smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)

# Plot the training history
plotUtils.plot_training_history(smaller_history)

# Create a bigger model
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=[num_words]),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer="adam",
                     loss="binary_crossentropy",
                     metrics=["accuracy", "binary_crossentropy"])

bigger_model.summary()

# Fit the bigger model
bigger_history = bigger_model.fit(train_data,
                                  train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

# Plot the training history
plotUtils.plot_training_history(bigger_history)

# Create a model with L2 weights regularization
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=[num_words]),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer="adam",
                 loss="binary_crossentropy",
                 metrics=["accuracy", "binary_crossentropy"])

# Fit the model
l2_history = l2_model.fit(train_data,
                          train_labels,
                          epochs=20,
                          batch_size=512,
                          validation_data=(test_data, test_labels),
                          verbose=2)

# Plot the training history
plotUtils.plot_training_history(l2_history)

# Create a model with some dropout layers
dropout_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=[num_words]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dropout_model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy", "binary_crossentropy"])

# Fit the model
dropout_history = dropout_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)

# Plot the training history
plotUtils.plot_training_history(dropout_history)

