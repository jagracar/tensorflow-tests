"""
Based on the following tutorial:
https://www.tensorflow.org/guide/feature_columns
"""

import numpy as np
import tensorflow as tf

# Create a numeric feature column
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength")

# Make the feature of type tf.float64
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength",
                                                          dtype=tf.float64)

# Create a bucketized feature column
numeric_feature_column = tf.feature_column.numeric_column("Year")
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column=numeric_feature_column,
    boundaries=[1960, 1980, 2000])

# Create a categorical identity feature column
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature',
    num_buckets=4)

# Create a categorical vocabulary feature column
vocabulary_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="my_feature_2",
    vocabulary_list=["kitchenware", "electronics", "sports"])

# Create a categorical hashed feature column
hashed_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
    key="some_feature",
    hash_bucket_size=100)
