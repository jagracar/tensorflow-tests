"""
Based on the following tutorial:
https://www.tensorflow.org/guide/premade_estimators
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# Download the iris datasets
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

# Read the train and test datasets
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop("Species")
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
test_x, test_y = test, test.pop("Species")


# Define the input functions
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    
    # Return the dataset.
    return dataset


# Create the feature columns
my_feature_columns = []

for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# Train the model
classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y, 100),
    steps=1000)

# Evaluate the model
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, 100))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']

predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size=100))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    
    print(template.format(SPECIES[class_id], 100 * probability, expec))
