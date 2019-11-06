"""
Some utility methods to do statistical modifications to input data sets and
ML results.

Based on some of the TensorFlow tutorials:
https://www.tensorflow.org/tutorials

Created on: 10/27/19
Author: Javier Gracia Carpio (jagracar@gmail.com)
"""

import numpy as np


def normalize_data(data, columns_statistics):
    """Normalizes a Pandas data frame to take values around -1 and 1.

    Parameters
    ----------
    data: object
        The Pandas data frame to normalize.
    columns_statistics: object
        The data frame columns statistics.

    Returns
    -------
    object
        The normalized Pandas data frame.

    """
    return (data - columns_statistics["mean"]) / columns_statistics["std"]


def create_multi_hot_encoding(sequences, dimension):
    """Creates a multi-hot-encoding array from a list of numeric sequences.

    Parameters
    ----------
    sequences: object
        A list of numeric sequences.
    dimension: int
        The multi-hot-encoding dimension.

    Returns
    -------
    object
        A numpy array with the multi-hot-encoding data.

    """
    # Initialize the multi-hot-encoding array
    multi_hot_encoding = np.zeros((len(sequences), dimension))

    # Fill the array with ones at the sequence indices
    for i, indices in enumerate(sequences):
        multi_hot_encoding[i, indices] = 1.0

    return multi_hot_encoding
