"""
A collection of TensorFlow Callback subclasses.

Based on some of the TensorFlow tutorials:
https://www.tensorflow.org/tutorials

Created on: 10/31/19
Author: Javier Gracia Carpio (jagracar@gmail.com)
"""

from tensorflow.keras.callbacks import Callback


class PrintDot(Callback):
    """A Callback subclass that prints a dot per training epoch.

    """

    def __init__(self, n_epochs):
        """Class constructor.

        Parameters
        ----------
        n_epochs: int
            The total number of epochs that are used to fit the model.

        Returns
        -------
        object
            A PrintDot callback instance.

        """
        # Use the super class constructor
        super().__init__()

        # Save the total number of epochs
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs):
        """Prints a dot per epoch.

        Parameters
        ----------
        epoch: int
            The current epoch number.
        logs: object
            The logs dictionary.

        """
        # Start a new line each time 5% of the epochs has been processed
        step = self.n_epochs // 20

        if epoch != 0 and epoch % step == 0:
            print("")

        # Print a dot
        print(".", end="")
