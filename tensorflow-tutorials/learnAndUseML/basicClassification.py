"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from utils import plotUtils

# Load the fashion MNIST data set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Inspect the data set
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Possible label values:", np.unique(train_labels))
print("Images minimum and maximum values:", train_images.min(), train_images.max())

# Translate the labels to their respective class names
class_names = np.array(["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])
train_classes = class_names[train_labels]
test_classes = class_names[test_labels]

# Plot the first image
plotUtils.plot_image(train_images[0], train_classes[0])

# Change the images to take values between 0 and 1 
train_images = train_images / 255
test_images = test_images / 255

# Plot the first 25 images
plotUtils.plot_images(train_images, train_classes, rows=5, columns=5)

# Define the model that we are going to use
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model using the train data set
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model using the test data set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)

# Predict the labels of the test images
predictions = model.predict(test_images)
print("Predictions shape:", predictions.shape)

# Plot the prediction for the first test image
index = 0
plotUtils.plot_prediction(predictions[index], test_images[index], test_classes[index], class_names)
