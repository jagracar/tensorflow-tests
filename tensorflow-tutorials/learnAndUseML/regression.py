"""
Based on the following tutorial:
https://www.tensorflow.org/tutorials/keras/regression
"""

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
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

# Split the data into a train and a test set
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

# Normalize the data sets
normalized_train_dataset = statsUtils.normalize_data(train_data, columns_statistics)
normalized_test_dataset = statsUtils.normalize_data(test_data, columns_statistics)


# Define and compile the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss="mean_squared_error",
                  metrics=["mean_absolute_error", "mean_squared_error"])

    return model


model = build_model()

# Print the model summary
model.summary()

# Train the model
EPOCHS = 1000
history = model.fit(normalized_train_dataset, train_labels,
                    epochs=EPOCHS, validation_split=0.2, verbose=0)

# Plot the training history
plotUtils.plot_training_history(history)

# Let"s build a new model to stop the training when there is not improvement with the validation data
model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(normalized_train_dataset, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop])

plotUtils.plot_training_history(history)

# Let's now evaluate the model on the test data
loss, mae, mse = model.evaluate(normalized_test_dataset, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# Predict the MPG values for the test data
test_predictions = model.predict(normalized_test_dataset).flatten()

# Compare the predictions with the real values
plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show(block=False)

# Plot the error distribution
error = test_predictions - test_labels

plt.figure()
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show(block=False)

