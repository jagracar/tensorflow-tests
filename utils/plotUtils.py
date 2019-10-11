"""
Some utility methods to plot input data sets and ML results.

Created on: 10/05/19
Author: Javier Gracia Carpio (jagracar@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_image(image, image_class):
    # Choose the color map to use
    color_map = plt.get_cmap("binary")

    # Plot the image
    plt.figure()
    plt.imshow(image, cmap=color_map)
    plt.grid(False)
    plt.title(image_class)
    plt.colorbar()
    plt.show(block=False)


def plot_images(images, image_classes, rows=5, columns=5, image_separation=0.3):
    # Set the figure dimensions
    image_box_size = 1.2
    margin_left = 0.2
    margin_right = 0.2
    margin_bottom = 0.2
    margin_top = 0.2 + image_separation
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
            ax.imshow(images[i], cmap=plt.get_cmap("binary"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(image_classes[i], fontsize=8)

    plt.show(block=False)


def plot_prediction(prediction, image, true_class, class_names):
    # Get the predicted image class and it's percentage
    predicted_label = np.argmax(prediction)
    predicted_class = class_names[predicted_label]
    percentage = 100 * prediction[predicted_label]

    # Get the true label
    true_label = np.where(class_names == true_class)[0][0]

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
    ax_0.set_title("%s %.0f%% (%s)" % (predicted_class, percentage, true_class))

    # Add the histogram
    ax_1 = plt.axes([(margin_left + image_box_size + image_separation) / figure_width,
                     margin_bottom / figure_height,
                     histogram_width / figure_width,
                     image_box_size / figure_height])
    n_bars = len(class_names)
    bars = ax_1.bar(range(n_bars), prediction, color="grey")
    bars[predicted_label].set_color("red")
    bars[true_label].set_color("blue")
    ax_1.set_xticks(range(n_bars))
    ax_1.set_xticklabels(class_names, rotation=45)
    ax_1.set_ylim([0, 1])
    ax_1.set_yticks([])
    ax_1.set_title("Model predictions")

    plt.show(block=False)
