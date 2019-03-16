"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/eager/eager_basics
"""

import time
import tempfile
import numpy as np
import tensorflow as tf

# Enable Eager execution
tf.enable_eager_execution()

# Print the result of some tensor operations
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))
print(tf.square(2) + tf.square(3))
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

# There is good compatibility between tensors and numpy arrays
ndarray = np.ones([3, 3])
tensor = tf.multiply(ndarray, 42)
print(tensor.numpy())

# One can check if a tensor resides in the CPU or the GPU memory
x = tf.random_uniform([3, 3])
print("Is there a GPU available: ", tf.test.is_gpu_available())
print("Is the Tensor on GPU #0:  ", x.device.endswith('GPU:0'))


# One can select the device where the tensors should be placed
def time_matmul(x):
    start = time.time()
    
    for loop in range(100):
        tf.matmul(x, x)
    
    result = time.time() - start
    print("100 loops: {:0.2f}ms".format(1000 * result))


with tf.device("CPU:0"):
    print("On CPU:")
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        print("On GPU:")
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)

# Create a dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3
  """)

# Read the file
ds_file = tf.data.TextLineDataset(filename)

# Apply some transformations
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

# Iterate over the datasets batches
for x in ds_tensors:
    print(x)

for x in ds_file:
    print(x)
