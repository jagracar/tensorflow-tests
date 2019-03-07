"""
Based on the following tutorial:
https://www.tensorflow.org/guide/saved_model
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

# Create some variables
v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)
inc_v1 = v1.assign(v1 + 1)
dec_v2 = v2.assign(v2 - 1)

# Add an operation to initialize the variables
init_op = tf.global_variables_initializer()

# Add operations to save and restore all the variables
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    inc_v1.op.run()
    dec_v2.op.run()
  
    # Save the variables to disk
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

# We can now restore the variables
tf.reset_default_graph()
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add operations to save and restore all the variables
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())

# Create some variables
tf.reset_default_graph()
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add operations to save and restore only v2 using the name v2
saver = tf.train.Saver({"v2": v2})

with tf.Session() as sess:
    v1.initializer.run()
    saver.restore(sess, "/tmp/model.ckpt")
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())

# Print all tensors in the checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# Print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

