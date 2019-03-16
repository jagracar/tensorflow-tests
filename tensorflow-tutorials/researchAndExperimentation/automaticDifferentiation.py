"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/eager/automatic_differentiation
"""

import numpy as np
import tensorflow as tf

# Enable Eager execution
tf.enable_eager_execution()

# Use GradientTape to calculate the gradient of a given operation
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz_dx = t.gradient(z, x)  # gradient((x + 3) * (x + 3)) = 2 * x + 2 * 3

for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0

# One needs to set the GradientTape persistent to use it several times
x = tf.constant(3.0)

with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y

dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
print(dz_dx, dy_dx)
del t


# Gradients can handle for and if statements
def f(x, y):
    output = 1.0
    
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    
    return output


def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    
    return t.gradient(out, x) 


x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0

# One can also compute higher order gradients
x = tf.Variable(1.0)

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x

    dy_dx = t2.gradient(y, x)

d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
