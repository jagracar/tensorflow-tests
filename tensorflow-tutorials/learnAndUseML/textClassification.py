"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/text_classification
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from utils import plotUtils

# Load the IMDB data set
(train_data, test_data), info = tfds.load("imdb_reviews/subwords8k",
                                          split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                          as_supervised=True,
                                          with_info=True)

# Get the text encoder and print the number of words in the vocabulary
encoder = info.features["text"].encoder
print ("Vocabulary size: %s" % encoder.vocab_size)

# Let's encode and decode a simple example
original_string = "Hello TensorFlow."
encoded_string = encoder.encode(original_string)
print("Encoded string is %s" % encoded_string)
print("The original string: %s" % encoder.decode(encoded_string))

for code in encoded_string:
    print("%s --> %s" % (code, encoder.decode([code])))

# Inspect the first 3 examples in the train data set
for train_example, train_label in train_data.take(3):
    print("Encoded text: %s" % train_example[:20].numpy())
    print("Decoded text: %s" % encoder.decode(train_example.numpy()))
    print("Label: %s" % train_label.numpy())
    print("\n")

# Prepare the data for training creating batches of 32 reviews
# with constant length (short reviews will be padded with zeros)
buffer_size = 1000
train_batches = (train_data.shuffle(buffer_size).padded_batch(32, train_data.output_shapes))
test_batches = (test_data.padded_batch(32, train_data.output_shapes))

for example_batch, label_batch in train_batches.take(2):
    print("Batch shape: %s" % example_batch.shape)
    print("Label shape: %s" % label_batch.shape)

# Define the model that we are going to use
model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    # keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model using the train batches
history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

# Plot the training history
plotUtils.plot_training_history(history)

# Evaluate the model using the test batches
test_loss, test_accuracy = model.evaluate(test_batches)
print("Test accuracy:", test_accuracy)

# Predict the labels of the test batches
predictions = model.predict(test_batches)
print("Predictions shape:", predictions.shape)
