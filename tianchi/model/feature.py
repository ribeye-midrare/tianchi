# Feature engineering

import tensorflow as tf

vocab_dir = '../data/feature_set/'

def build_feature_columns():
  float_features  = {'1814', '1815', '1850', '190', '191', '192', '2403', '2404', '2405', '10004'}
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
