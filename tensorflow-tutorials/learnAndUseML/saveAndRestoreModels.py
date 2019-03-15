"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/save_and_restore_models
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Select only the first 1000 samples and reshape and normalize the images
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# Define the model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
    
    return model


model = create_model()
model.summary()

# Create a checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Fit the model and save the weights
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

# Create a new untrained model
model = create_model()

# Evaluate the untrained model
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Load the weights from the checkpoints
model.load_weights(checkpoint_path)

# Evaluate the model again
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Create a new checkpoint callback with more options
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=5)

# Create a new model and save the initial weights
model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))

# Fit the model and save the weights every 5 epochs
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# Load the latest checkpoint and use it with a new model
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# One can also save the weights manually
model.save_weights('training_3/my_checkpoint')
model = create_model()
model.load_weights('training_3/my_checkpoint')
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# One can save complete models as an HDF5 file
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')

# Recreate the exact same model
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# And one can also use the TF saved_models module
model = create_model()
model.fit(train_images, train_labels, epochs=5)
saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

# Recreate the exact same model
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()

# The model needs to be compiled again
new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
