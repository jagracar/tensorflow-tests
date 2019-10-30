"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/regression
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from utils import plotUtils
from utils import statsUtils

# Download the Auto MPG data set and save it into a file
file_name = keras.utils.get_file(
    "auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(file_name)

# Load the CSV file using Pandas
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight",
                "Acceleration", "Model Year", "Origin"]
raw_data = pd.read_csv(file_name, names=column_names, na_values="?",
                       comment="\t", sep=" ", skipinitialspace=True)
car_names = pd.read_csv(file_name, names=["temp", "Names"], sep="\t",
                        skipinitialspace=True)["Names"]

# Make a copy for pre-processing and print the last rows of the data set
data = raw_data.copy()
data.tail()

# Check the number of NaN values in each column
print(data.isna().sum())

# Remove rows with NaN values
data = data.dropna()

# Convert the (categorical) Origin column to one-hot (numeric)
origin = data.pop("Origin")
data["USA"] = np.array(origin == 1, dtype=np.float)
data["Europe"] = np.array(origin == 2, dtype=np.float)
data["Japan"] = np.array(origin == 3, dtype=np.float)

# Split the data into a train set and a test set
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Have a quick view to the general trends
sns.pairplot(train_data[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show(block=False)

# Separate the features from the labels
train_labels = train_data.pop("MPG")
test_labels = test_data.pop("MPG")

# Calculate the columns statistics using only the train data set
columns_statistics = train_data.describe().transpose()
print(columns_statistics)

# Normalize the data sets to take values mostly between -1 and 1
normalized_train_data = statsUtils.normalize_data(train_data, columns_statistics)
normalized_test_data = statsUtils.normalize_data(test_data, columns_statistics)


# Create a function to build the model in one go
def build_model():
    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss="mean_squared_error",
                  metrics=["mean_absolute_error", "mean_squared_error"])

    return model


# Build the model
model = build_model()

# Print the model summary
model.summary()

# Train the model using the normalized train data set
history = model.fit(normalized_train_data, train_labels, epochs=1000,
                    validation_split=0.2, verbose=0)

# Plot the training history
plotUtils.plot_training_history(history)

# Build the model again
model = build_model()

# This time we will stop the training when the there is no
# improvement with the validation data
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
history = model.fit(normalized_train_data, train_labels, epochs=1000,
                    validation_split=0.2, verbose=0, callbacks=[early_stop])

# Plot the training history
plotUtils.plot_training_history(history)

# Evaluate the model using the normalized test data set
loss, mae, mse = model.evaluate(normalized_test_data, test_labels, verbose=0)
print("Test set Mean Absolute Error: %5.2f MPG" % mae)

# Predict the MPG values for the normalized test data set
test_predictions = model.predict(normalized_test_data).flatten()

# Plot the predictions
plotUtils.plot_regression_predictions(test_predictions, test_labels)
