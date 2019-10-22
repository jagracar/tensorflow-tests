"""
Some utility methods to plot input data sets and ML results.

Created on: 10/05/19
Author: Javier Gracia Carpio (jagracar@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt


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
    plt.figure()
    plt.imshow(image, cmap=color_map)
    plt.title(image_class)
    plt.colorbar()
    plt.show(block=False)


def plot_images(images, image_classes, rows=5, columns=5):
    """Plots a set of images together with their class names.

    Parameters
    ----------
    images: object
        A 3D numpy array with the images pixel values.
    image_classes: object
        A numpy array with the images class names.
    rows: int, optional
        The number of image rows in the plot. The total number of images
        displayed is rows * columns. Default is 5.
    columns: int, optional
        The number of image columns in the plot. The total number of images
        displayed is rows * columns. Default is 5.

    """
    # Choose the color map to use
    color_map = plt.get_cmap("binary")

    # Set the figure dimensions
    image_box_size = 1.2
    image_separation = 0.3
    margin_left = 0.2
    margin_right = 0.2
    margin_bottom = 0.2
    margin_top = 0.4
    figure_width = (margin_left + columns * image_box_size + 
                    (columns - 1) * image_separation + margin_right)
    figure_height = (margin_bottom + rows * image_box_size + 
                     (rows - 1) * image_separation + margin_top)
    figure_size = (figure_width, figure_height)

    # Initialize the figure
    _, ax_list = plt.subplots(rows, columns, figsize=figure_size)

    # Set the subplot layout
    plt.subplots_adjust(
        left=margin_left / figure_width,
        right=(figure_width - margin_right) / figure_width,
        bottom=margin_bottom / figure_height,
        top=(figure_height - margin_top) / figure_height,
        wspace=image_separation / image_box_size,
        hspace=image_separation / image_box_size)

    # Add the images to the figure
    for i, ax in enumerate(ax_list.ravel()):
        if i < len(images):
            # Display the image
            ax.imshow(images[i], cmap=color_map)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(image_classes[i], fontsize=8)

    plt.show(block=False)


def plot_prediction(prediction, image, image_class, class_names):
    """Plots the prediction from an image classification model.

    Parameters
    ----------
    prediction: object
        A numpy array with the model prediction probabilities.
    image: object
        A 2D numpy array with the image pixel values.
    image_class: str
        The image class name.
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
    margin_left = 0.2
    margin_right = 0.2
    margin_bottom = 0.8
    margin_top = 0.4
    figure_width = (margin_left + image_box_size + image_separation + 
                    histogram_width + margin_right)
    figure_height = (margin_bottom + image_box_size + margin_top)
    figure_size = (figure_width, figure_height)

    # Initialize the figure
    plt.figure(figsize=figure_size)

    # Add the image
    ax_0 = plt.axes([margin_left / figure_width,
                     margin_bottom / figure_height,
                     image_box_size / figure_width,
                     image_box_size / figure_height])
    ax_0.imshow(image, cmap=color_map)
    ax_0.set_xticks([])
    ax_0.set_yticks([])
    ax_0.set_title("%s %.0f%% (%s)" % (predicted_class, percentage, image_class))

    # Get the true image label
    image_label = np.where(class_names == image_class)[0][0]

    # Add the histogram
    ax_1 = plt.axes([(margin_left + image_box_size + image_separation) / figure_width,
                     margin_bottom / figure_height,
                     histogram_width / figure_width,
                     image_box_size / figure_height])
    x_ticks = np.arange(len(class_names))
    bars = ax_1.bar(x_ticks, prediction, color="grey")
    bars[predicted_label].set_color("red")
    bars[image_label].set_color("blue")
    ax_1.set_xticks(x_ticks)
    ax_1.set_xticklabels(class_names, rotation=45)
    ax_1.set_ylim([0, 1])
    ax_1.set_yticks([])
    ax_1.set_title("Model predictions")

    plt.show(block=False)


def plot_predictions(predictions, images, image_classes, class_names, rows=5, columns=3):
    """Plots the predictions from an image classification model.

    Parameters
    ----------
    predictions: object
        A 2D numpy array with the model predictions probabilities.
    images: object
        A 3D numpy array with the images pixel values.
    image_classes: object
        A numpy array with the images class names.
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

    # Set the figure dimensions
    image_box_size = 1.2
    image_separation = 0.3
    margin_left = 0.2
    margin_right = 0.2
    margin_bottom = 0.2
    margin_top = 0.4
    figure_width = (margin_left + 2 * columns * image_box_size + 
                    (2 * columns - 1) * image_separation + margin_right)
    figure_height = (margin_bottom + rows * image_box_size + 
                     (rows - 1) * image_separation + margin_top)
    figure_size = (figure_width, figure_height)

    # Initialize the figure
    _, ax_list = plt.subplots(rows, 2 * columns, figsize=figure_size)

    # Set the subplot layout
    plt.subplots_adjust(
        left=margin_left / figure_width,
        right=(figure_width - margin_right) / figure_width,
        bottom=margin_bottom / figure_height,
        top=(figure_height - margin_top) / figure_height,
        wspace=image_separation / image_box_size,
        hspace=image_separation / image_box_size)

    # Add the images to the figure
    for i, ax in enumerate(ax_list.ravel()):
        # Get the image index
        index = i // 2

        if index < len(images):
            if i % 2 == 0:
                # Add the image
                ax.imshow(images[index], cmap=color_map)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("%s %.0f%% (%s)" % (
                    predicted_classes[index], percentages[index], image_classes[index]), fontsize=8)
            else:
                # Get the true image label
                image_label = np.where(class_names == image_classes[index])[0][0]

                # Add the histogram
                x_ticks = np.arange(len(class_names))
                bars = ax.bar(x_ticks, predictions[index], color="grey")
                bars[predicted_labels[index]].set_color("red")
                bars[image_label].set_color("blue")
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticks, fontdict={"fontsize" : 8})
                ax.set_ylim([0, 1])
                ax.set_yticks([])

    plt.show(block=False)
