"""
Some utility methods to do some statistical modifications to input data sets
and ML results.

Based on some of the TensorFlow tutorials:
https://www.tensorflow.org/tutorials/

Created on: 10/27/19
Author: Javier Gracia Carpio (jagracar@gmail.com)
"""

import numpy as np


def normalize_data(data, columns_statistics):
    return (data - columns_statistics["mean"]) / columns_statistics["std"]

