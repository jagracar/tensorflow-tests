"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/save_and_load
"""

import os.path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

# Load the fashion MNIST data set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# We will consider only the first 1000 samples
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000]
test_images = test_images[:1000]

# Scale the images to take values between 0 and 1
train_images = train_images / 255
test_images = test_images / 255


# Create a function to build the model in one go
def build_model():
    # Define the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Compile the model
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


# Build the model
model = build_model()

# Print the model summary
model.summary()

# Create a checkpoint callback that saves the model's weights
checkpoint_path = "out/training_1/cp.ckpt"
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

# Train the model using the normalized train data set
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

# Build a new untrained model
model = build_model()

# Evaluate the untrained model
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Untrained model accuracy: %.1f%%" % (100 * acc))

# Load the weights from the checkpoints
model.load_weights(checkpoint_path)

# Evaluate the model again
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Restored model accuracy: %.1f%%" % (100 * acc))

# Create a new checkpoint callback with more options
checkpoint_path = "out/training_2/cp-{epoch:04d}.ckpt"
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              period=5,
                                              verbose=1)

# Build a new model and save the initial weights
model = build_model()
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model and save the weights every 5 epochs
model.fit(train_images, train_labels, epochs=50,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback],
          verbose=0)

# Load the latest checkpoint and use it with a new model
latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
model = build_model()
model.load_weights(latest)

# Evaluate the model again
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Restored model accuracy: %.1f%%" % (100 * acc))

# One can also save the weights manually
checkpoint_path = "out/training_3/my_checkpoint"
model.save_weights(checkpoint_path)
model = build_model()
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Restored model accuracy: %.1f%%" % (100 * acc))

# One can save complete models as an HDF5 file
model = build_model()
model.fit(train_images, train_labels, epochs=5)
model_path = "out/my_model.h5"
model.save(model_path)

# Recreate the exact same model
new_model = keras.models.load_model(model_path)
new_model.summary()
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Restored model accuracy: %.1f%%" % (100 * acc))

# And one can also use the TF SavedModel format
model = build_model()
model.fit(train_images, train_labels, epochs=5)
model_path = "out/my_model"
model.save(model_path)

# Recreate the exact same model
new_model = keras.models.load_model(model_path)
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=0)
print("Restored model accuracy: %.1f%%" % (100 * acc))
