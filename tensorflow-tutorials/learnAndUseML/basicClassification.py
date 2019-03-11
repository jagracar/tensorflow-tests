"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/basic_classification
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load the fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Check the dataset shapes
print(train_images.shape)
print(train_labels.shape)

# Check how many labels we have (from 0 to 9)
print(np.unique(train_labels))

# Check the images gray color ranges (0 and 255)
print(train_images.min(), train_images.max())

# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot the first image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show(block=False)

# Let's scale the image rescale the image values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Plot the first 25 images
plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show(block=False)

# Define the model that we are going to use
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model using the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Predict the labels of the test images
predictions = model.predict(test_images)
print(predictions.shape)

# Get the predicted labels
predicted_labels = np.argmax(predictions, axis=1)


# Lets visualize the model results
def plot_image(i, predictions, predicted_labels, true_labels, images):  
    color = 'blue' if predicted_labels[i] == true_labels[i] else 'red'
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)  
    plt.imshow(images[i] , cmap=plt.cm.binary)
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_labels[i]],
                                         100 * np.max(predictions[i]),
                                         class_names[true_labels[i]]),
                                         color=color)


def plot_value_array(i, predictions, predicted_labels, true_labels):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.ylim([0, 1])  
    bars = plt.bar(range(10), predictions[i], color="#777777")
    bars[predicted_labels[i]].set_color('red')
    bars[true_labels[i]].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, predicted_labels, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, predicted_labels, test_labels)

plt.show(block=False)

# We can predict the label of one of the test images
img = test_images[0]
img = img[np.newaxis]
predictions = model.predict(img)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure()
plot_value_array(0, predictions, predicted_labels, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show(block=False)
