"""
Based on the following tutorial:
https://www.tensorflow.org/guide/eager
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Enable eager execution
tf.enable_eager_execution()

# Check that it worked
tf.executing_eagerly() 

# Operations take place without a session
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)

b = tf.add(a, 1)
print(b)

print(a * b)

c = np.multiply(a, b)
print(c)

print(a.numpy())


# Create a layer subclass
class MySimpleLayer(tf.keras.layers.Layer):
    
    def __init__(self, output_units):
        super(MySimpleLayer, self).__init__()
        self.output_units = output_units
    
    def build(self, input_shape):
        self.kernel = self.add_variable(
            "kernel", [input_shape[-1], self.output_units])
    
    def call(self, input):
        return tf.matmul(input, self.kernel)


# Create a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,)),
    tf.keras.layers.Dense(10)
])


# Or define your own model subclass
class MNISTModel(tf.keras.Model):
    
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=10)
    
    def call(self, input):
        """Run the model."""
        result = self.dense1(input)
        result = self.dense2(result)
        result = self.dense2(result)
        return result


model = MNISTModel()

# Fetch and format the MNIST data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
     tf.cast(mnist_labels, tf.int64)))

dataset = dataset.shuffle(1000).batch(32)

# Build the model
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

# Inspect the model output before training
for images, labels in dataset.take(1):
    print("Logits: ", mnist_model(images[0:1]).numpy())

# Train the model
optimizer = tf.train.AdamOptimizer()
loss_history = []

for (batch, (images, labels)) in enumerate(dataset.take(400)):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    
    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables),
                              global_step=tf.train.get_or_create_global_step())

# Plot the loss evolution
plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()


# Define a linear model
class Model(tf.keras.Model):
    
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(5., name='weight')
        self.B = tf.Variable(10., name='bias')
    
    def call(self, inputs):
        return inputs * self.W + self.B


# Create a toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise


# Define the loss function to be optimized
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))


# Define the function that computes the gradients
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])


# Fit the model to the points
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

for i in range(300):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                              global_step=tf.train.get_or_create_global_step())
    if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

