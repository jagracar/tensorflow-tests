"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from utils import plotUtils

# Load the IMDB data set
((train_data, validation_data), test_data) = tfds.load(
    name="imdb_reviews",
    split=(tfds.Split.TRAIN.subsplit([6, 4]), tfds.Split.TEST),
    as_supervised=True)

# Inspect the first 3 examples in the train data set
for train_example, train_label in train_data.take(3):
    print("Review text: %s" % train_example.numpy())
    print("Label: %s" % train_label.numpy())
    print("\n")

# We will use a pre-trained text embedding model from TensorFlow Hub
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string,
                           trainable=True)

# Apply the embedding on the first 3 examples in the train data set
for train_example, train_label in train_data.take(3):
    print("Review text: %s" % train_example.numpy())
    print("Embedding tensor: %s" % hub_layer([train_example]))

# Define the model that we are going to use
model = keras.Sequential([
    hub_layer,
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model using batches with 512 examples each
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Plot the training history
plotUtils.plot_training_history(history)

# Evaluate the model using the test batches
test_loss, test_accuracy = model.evaluate(test_data.batch(512), verbose=0)
print("Test accuracy:", test_accuracy)

# Predict the labels of the test batches
predictions = model.predict(test_data.batch(512))
print("Predictions shape:", predictions.shape)
