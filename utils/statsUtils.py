"""
Some utility methods to do statistical modifications to input data sets and
ML results.

Based on some of the TensorFlow tutorials:
https://www.tensorflow.org/tutorials

Created on: 10/27/19
Author: Javier Gracia Carpio (jagracar@gmail.com)
"""


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
