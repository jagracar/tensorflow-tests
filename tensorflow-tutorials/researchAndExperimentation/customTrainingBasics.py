"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/eager/custom_training
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Enable Eager execution
tf.enable_eager_execution()


# Define a linear model (y = x * W + b)
class Model(object):

    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
    
    def __call__(self, x):
        return self.W * x + self.b


model = Model()
assert model(3.0).numpy() == 15.0


# Define the loss function
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


# Create some data for training
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

# Plot the data trend together with the untrained model predictions
plt.figure()
plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show(block=False)

# Calculate the loss value
print('Current loss: ', loss(model(inputs), outputs).numpy())


# Define the train method
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


# Train the model and save the intermediate results
model = Model()

Ws, bs = [], []
epochs = range(10)

for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % 
          (epoch, Ws[-1], bs[-1], current_loss))

# Plot the results
plt.figure()
plt.plot(epochs, Ws, 'r')
plt.plot(epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--')
plt.plot([TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show(block=False)
