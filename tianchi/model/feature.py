# Feature engineering

import tensorflow as tf

vocab_dir = '../data/feature_set/'

def _build_dense_features():
  float_features  = {'1814', '1815', '1850', '190', '191', '192', '2403', '2404', '2405', '10004'}
  bucketized_feature = []

  boundaries = {
    '1814': [10.10, 12.00, 13.20, 14.71, 16.00, 17.00, 18.00, 19.35, 21.00, 22.00, 24.00, 26.00, 28.00, 30.00, 33.00, 37.00, 42.00, 50.90, 66.69, 298.00],
    '1815': [14.00, 15.00, 16.00, 17.00, 18.00, 18.40, 19.00, 20.00, 20.60, 21.10, 22.00, 23.00, 24.00, 25.00, 26.00, 28.00, 30.00, 33.00, 39.60, 150.00],
    '1850': [4.11, 4.29, 4.43, 4.54, 4.63, 4.72, 4.80, 4.88, 4.96, 5.03, 5.11, 5.20, 5.29, 5.39, 5.50, 5.65, 5.84, 6.15, 7.08, 18.70],
    '190': [46.00, 49.50, 52.10, 54.90, 57.00, 59.11, 61.90, 64.00, 66.48, 69.00, 71.21, 73.80, 76.00, 78.63, 81.21, 84.50, 88.38, 93.00, 100.70, 258.00],
    '191': [197.90, 219.38, 236.10, 250.57, 263.50, 275.60, 288.54, 300.40, 312.11, 324.42, 337.00, 349.00, 362.70, 377.63, 394.00, 411.00, 430.00, 459.00, 500.80, 732.00],
    '192': [7.50, 8.95, 10.10, 11.26, 12.33, 13.50, 15.00, 16.80, 19.90, 61.90],
    '2403': [49.30, 52.60, 55.00, 57.60, 59.90, 62.00, 64.30, 66.80, 69.00, 71.80, 74.20, 77.30, 81.10, 87.20, 150.10],
    '2404': [152.50, 155.50, 157.50, 159.40, 161.00, 162.50, 164.00, 166.00, 167.50, 169.00, 171.00, 173.00, 175.00, 178.00, 195.70],
    '2405': [20.00, 21.30, 22.30, 23.20, 24.10, 25.00, 26.00, 27.20, 28.90, 49.30],
    '10004': [3.35, 3.78, 4.10, 4.40, 4.70, 5.00, 5.36, 5.81, 6.48, 17.80]
  }

  for feature_name in float_features:
    nm_column = tf.feature_column.numeric_column(feature_name, default_value=0.0)
    boundary = boundaries[feature_name]
    bucketized_feature.append(tf.feature_column.bucketized_column(nm_column, boundary))

  return bucketized_feature

def _build_sparse_features():
  sparse_features = {'2302', '1840', '3190', '3191', '3192', '3193', '3195', '3196', '3197',
                     '3429', '3430', '3730', '300005', '0407', '0420', '0421', '0423', '0426', '0430',
                     '0431', '0901', '100010'}
  feature_columns = []

  for sparse_feature in sparse_features:
    categorical_feature = tf.feature_column.categorical_column_with_vocabulary_file(
      sparse_feature,
      vocab_dir + sparse_feature,
      num_oov_buckets=5,
      dtype=tf.string)
    feature_columns.append(tf.feature_column.embedding_column(categorical_feature, 16))

  # for float_feature in float_features:
  #   feature_columns.append(tf.feature_column.numeric_column(float_feature))

  return feature_columns

def build_feature_columns():
  return _build_sparse_features() + _build_dense_features()
