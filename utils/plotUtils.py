"""
Some utility methods to plot input data sets and ML results.

Based on some of the TensorFlow tutorials:
https://www.tensorflow.org/tutorials

Created on: 10/05/19
Author: Javier Gracia Carpio (jagracar@gmail.com)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_multipanel_figure(rows, columns, panel_width, panel_height,
                             panel_separation, margins):
    """Creates a multi-panel figure.

    Parameters
    ----------
    rows: int
        The number of panel rows.
    columns: int
        The number of panel columns.
    panel_width: float
        The panel width.
    panel_height: float
        The panel height.
    panel_separation: float
        The separation between panels.
    margins: list
        A python list with the margin sizes in the following sequence: left,
        top, right, bottom.

    Returns
    -------
    tuple
        A python tuple with the figure and the panels axes list.

    """
    # Extract the margins information
    left_margin = margins[0]
    top_margin = margins[1]
    right_margin = margins[2]
    bottom_margin = margins[3]

    # Calculate the figure dimensions
    figure_width = (left_margin + columns * panel_width + 
                    (columns - 1) * panel_separation + right_margin)
    figure_height = (bottom_margin + rows * panel_height + 
                     (rows - 1) * panel_separation + top_margin)
    figure_size = (figure_width, figure_height)

    # Initialize the figure
    fig, axes_list = plt.subplots(rows, columns, figsize=figure_size)

    # Set the subplot layout
    plt.subplots_adjust(
        left=left_margin / figure_width,
        right=(figure_width - right_margin) / figure_width,
        bottom=bottom_margin / figure_height,
        top=(figure_height - top_margin) / figure_height,
        wspace=panel_separation / panel_width,
        hspace=panel_separation / panel_height)

    return fig, axes_list


def plot_image(image, image_class):
    """Plots an image together with its class name.

    Parameters
    ----------
    image: object
        A 2D numpy array with the image pixel values.
    image_class: str
        The image class name.

    """
    # Choose the color map to use
    color_map = plt.get_cmap("binary")

    # Plot the image
    fig = plt.figure()
    plt.imshow(image, cmap=color_map)
    plt.title(image_class)
    plt.colorbar()
    fig.show()


def plot_images(images, image_classes, rows=5, columns=5):
    """Plots a set of images together with their class names.

    Parameters
    ----------
    images: object
        A 3D numpy array with the images pixel values.
    image_classes: object
        A numpy array with the images classes.
    rows: int, optional
        The number of image rows in the plot. The total number of images
        displayed is rows * columns. Default is 5.
    columns: int, optional
        The number of image columns in the plot. The total number of images
        displayed is rows * columns. Default is 5.

    """
    # Choose the color map to use
    color_map = plt.get_cmap("binary")

    # Create the multi-panel figure
    fig, axes_list = create_multipanel_figure(rows, columns,
                                              panel_width=1.2,
                                              panel_height=1.2,
                                              panel_separation=0.3,
                                              margins=[0.2, 0.4, 0.2, 0.2])

    # Add the images to the figure
    for i, axes in enumerate(axes_list.ravel()):
        if i < len(images):
            # Display the image
            axes.imshow(images[i], cmap=color_map)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_title(image_classes[i], fontsize=8)

    fig.show()


def plot_image_classification_prediction(prediction, image, image_class,
                                         class_names):
    """Plots the prediction from an image classification model.

    Parameters
    ----------
    prediction: object
        A numpy array with the model prediction probabilities.
    image: object
        A 2D numpy array with the image pixel values.
    image_class: str
        The image class.
    class_names: object
        A numpy array will all the possible class names.

    """
    # Get the predicted image class and its percentage
    predicted_label = np.argmax(prediction)
    predicted_class = class_names[predicted_label]
    percentage = 100 * np.max(prediction)

    # Choose the color map to use
    color_map = plt.get_cmap("binary")

    # Set the figure dimensions
    image_box_size = 3
    histogram_width = 4.5
    image_separation = 0.1
    left_margin = 0.2
    right_margin = 0.2
    bottom_margin = 0.8
    top_margin = 0.4
    figure_width = (left_margin + image_box_size + image_separation + 
                    histogram_width + right_margin)
    figure_height = (bottom_margin + image_box_size + top_margin)
    figure_size = (figure_width, figure_height)

    # Initialize the figure
    fig = plt.figure(figsize=figure_size)

    # Add the image
    axes_0 = plt.axes([left_margin / figure_width,
                       bottom_margin / figure_height,
                       image_box_size / figure_width,
                       image_box_size / figure_height])
    axes_0.imshow(image, cmap=color_map)
    axes_0.set_xticks([])
    axes_0.set_yticks([])
    axes_0.set_title("%s %.0f%% (%s)" % (predicted_class, percentage, image_class))

    # Get the true image label
    image_label = np.where(class_names == image_class)[0][0]

    # Add the histogram
    axes_1 = plt.axes([(left_margin + image_box_size + image_separation) / figure_width,
                       bottom_margin / figure_height,
                       histogram_width / figure_width,
                       image_box_size / figure_height])
    x_ticks = np.arange(len(class_names))
    bars = axes_1.bar(x_ticks, prediction, color="grey")
    bars[predicted_label].set_color("red")
    bars[image_label].set_color("blue")
    axes_1.set_xticks(x_ticks)
    axes_1.set_xticklabels(class_names, rotation=45)
    axes_1.set_ylim([0, 1])
    axes_1.set_yticks([])
    axes_1.set_title("Model predictions")

    fig.show()


def plot_image_classification_predictions(predictions, images, image_classes,
                                          class_names, rows=5, columns=3):
    """Plots the predictions from an image classification model.

    Parameters
    ----------
    predictions: object
        A 2D numpy array with the model predictions probabilities.
    images: object
        A 3D numpy array with the images pixel values.
    image_classes: object
        A numpy array with the images classes.
    class_names: object
        A numpy array will all the possible class names.
    rows: int, optional
        The number of image rows in the plot. The total number of images
        displayed is rows * columns. Default is 5.
    columns: int, optional
        The number of image columns in the plot. The total number of images
        displayed is rows * columns. Default is 3.

    """
    # Get the predicted image classes and their percentages
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_classes = class_names[predicted_labels]
    percentages = 100 * np.max(predictions, axis=1)

    # Choose the color map to use
    color_map = plt.get_cmap("binary")

    # Create the multi-panel figure
    fig, axes_list = create_multipanel_figure(rows, 2 * columns,
                                              panel_width=1.2,
                                              panel_height=1.2,
                                              panel_separation=0.3,
                                              margins=[0.2, 0.4, 0.2, 0.2])

    # Add the predictions to the figure
    for i, axes in enumerate(axes_list.ravel()):
        # Get the image index
        index = i // 2

        if index < len(images):
            if i % 2 == 0:
                # Add the image
                axes.imshow(images[index], cmap=color_map)
                axes.set_xticks([])
                axes.set_yticks([])
                axes.set_title("%s %.0f%% (%s)" % (
                    predicted_classes[index], percentages[index], image_classes[index]), fontsize=8)
            else:
                # Get the true image label
                image_label = np.where(class_names == image_classes[index])[0][0]

                # Add the histogram
                x_ticks = np.arange(len(class_names))
                bars = axes.bar(x_ticks, predictions[index], color="grey")
                bars[predicted_labels[index]].set_color("red")
                bars[image_label].set_color("blue")
                axes.set_xticks(x_ticks)
                axes.set_xticklabels(x_ticks, fontdict={"fontsize" : 8})
                axes.set_ylim([0, 1])
                axes.set_yticks([])

    fig.show()


def plot_regression_predictions(predictions, true_values):
    """Plots the predictions from a regression model.

    Parameters
    ----------
    predictions: object
        A numpy array with the model predictions.
    true_values: object
        A numpy array with the true values.

    """
    # Create the multi-panel figure
    fig, axes_list = create_multipanel_figure(1, 2,
                                              panel_width=4.0,
                                              panel_height=4.0,
                                              panel_separation=0.7,
                                              margins=[0.6, 0.4, 0.2, 0.6])

    # Add the predictions scatter plot
    axes_list[0].scatter(true_values, predictions)
    axes_list[0].set_xlabel("True values")
    axes_list[0].set_ylabel("Predictions")

    # Add a line with slope 1
    xlim = axes_list[0].get_xlim()
    ylim = axes_list[0].get_ylim()
    line_range = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    axes_list[0].plot(line_range, line_range, "--k", alpha=0.5, zorder=-1)

    # Add the predictions error histogram
    axes_list[1].hist(predictions - true_values, bins=25)
    axes_list[1].set_xlabel("Predictions - True values")
    axes_list[1].set_ylabel("Counts")

    fig.suptitle("Model prediction results")
    fig.show()


def plot_training_history(training_history):
    """Plots the model training history.

    Parameters
    ----------
    training_history: object
        The model training history returned by Keras model.fit().

    """
    # Create a Pandas data frame with the training history
    history = pd.DataFrame(training_history.history)

    # Get the metrics to plot
    metrics = [column for column in history if not column.startswith("val_")]

    # Create the multi-panel figure
    fig, axes_list = create_multipanel_figure(len(metrics), 1,
                                              panel_width=6,
                                              panel_height=2.5,
                                              panel_separation=0.4,
                                              margins=[0.7, 0.4, 0.2, 0.5])

    # Add all the metric histories to the figure
    epoch = np.array(training_history.epoch) + 1

    for i, axes in enumerate(axes_list.ravel()):
        # Plot the metric history
        y = history[metrics[i]]
        axes.plot(epoch, y, label="Training set")
        axes.set_ylabel(metrics[i].replace("_", " ").title())

        # Calculate the y axis maximum and minimum values
        y_std = y[len(y) // 5:].std()
        min_value = max(0, y.min() - y_std)
        max_value = min(y.median() + 10 * y_std, y.max() + 3 * y_std)
        axes.set_ylim([min_value, max_value])        

        # Check if the metric validation history is available
        metric_validation = "val_" + metrics[i]

        if metric_validation in history:
            # Plot the validation metric history
            y = history[metric_validation]
            axes.plot(epoch, y, label="Validation set")

            # Add the legend
            axes.legend(loc="upper right")

            # Update the plot maximum and minimum values
            y_std = y[len(y) // 5:].std()
            min_value = min(min_value, max(0, y.min() - y_std))
            max_value = max(max_value,
                            min(y.median() + 10 * y_std, y.max() + 3 * y_std))
            axes.set_ylim([min_value, max_value])

    axes_list[0].set_title("Training history")
    axes_list[-1].set_xlabel("Epoch")

    fig.show()
