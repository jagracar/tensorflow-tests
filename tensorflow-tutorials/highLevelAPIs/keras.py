"""
Based on the following tutorial:
https://www.tensorflow.org/guide/keras
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Initialize a sequential model
model = tf.keras.Sequential()

# Add a densely-connected layer with 64 units
model.add(layers.Dense(64, activation='relu', input_shape=(32,)))

# Add another dense layer
model.add(layers.Dense(64, activation='relu'))

# Add a softmax layer with 10 output units
model.add(layers.Dense(10, activation='softmax'))

# Compile the model setting the optimizer the loss function and the metrics
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Let's define some input data with the input and output model dimensions
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

# Create some validation data
val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

# Fit the model to the data
model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

# One can also use Datasets
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

# Now we can evaluate the model
model.evaluate(data, labels, batch_size=32)

# And we can use the model to predict the results
result = model.predict(data, batch_size=32)
print(result.shape)

# Create a placeholder tensor
inputs = tf.keras.Input(shape=(32,))

# Create a sequence of connected layers
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

# Instantiate the model
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# And train the model
model.fit(data, labels, batch_size=32, epochs=5)


# We can also create our specific model subclasses
class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
    
        # Define your layers here
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`)
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


# Create a new model instance
model = MyModel(num_classes=10)

# Compile the model
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(data, labels, batch_size=32, epochs=5)


# We can also create our own custom layers
class MyLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        
        # Make sure to call the `build` method at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Create a model using the custom layer
model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

# Compile the model
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(data, labels, batch_size=32, epochs=5)

# We can add callback functions to a model
callbacks = [
  # Interrupt training if val_loss stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to ./logs directory
  # tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))

# Now we can save the model weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Restore the model's state,
model.load_weights('./weights/my_model')

# We can also serialize the model (without weights)
model = tf.keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(32,)),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)

json_string = model.to_json()
print(json_string)

# We can recreate the model from the json string
fresh_model = tf.keras.models.model_from_json(json_string)

# We can save a complete model (structure + weights)
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(32,)),
  layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)

model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer
model = tf.keras.models.load_model('my_model.h5')
