# Reads training, eval, and test data, and analyze all features

from google.protobuf import text_format
import tensorflow as tf

train_filename = '../data/v3/train.tfrecords'
eval_filename  = '../data/v3/eval.tfrecords'
test_filename  = '../data/v3/test.tfrecords'
output_dir     = '../data/v3/feature_eval/'

# output format
# ../data/v3/feature_eval/0102.train
# ../data/v3/feature_eval/0102.eval
# ../data/v3/feature_eval/0102.test
# ../data/v3/feature_eval/0115.train
# ../data/v3/feature_eval/0115.eval
# ../data/v3/feature_eval/0115.test
#
# In each file, it is:
# summary: x examples, y unique values
# value_1, example_count
# value_2, example_count
# ...
# sorted by example count reversely


# TODO: put all features in features.py
float_features  = {'1814', '1815', '1850', '190', '191', '192', '2403', '2404', '2405', '10004'}
sparse_features = {'2302', '1840', '3190', '3191', '3192', '3193', '3195', '3196', '3197',
                   '3429', '3430', '3730', '300005', '0407', '0420', '0421', '0423', '0426', '0430',
                   '0431', '0901', '100010'}
nlp_features    = {'0102', '0115'}

def _read_tf_examples(file_name):
  record_iterator = tf.python_io.tf_record_iterator(file_name)
  examples = []
  for str_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(str_record)
    examples.append(example)
  return examples

def _write_stats(stats, output_file, total_examples):
  output = open(output_file, 'w+')
  output.write('Total Examples: {}, Unique Values: {}\n'.format(total_examples, len(stats)))
  for key, value in sorted(stats.items(), key=lambda x: x[1]):
    output.write('{}: {}\n'.format(value, key))

  output.close()

def _analyze_float_feature(examples, feature_name, output_file):
  stat = {}  # value: count
  for example in examples:
    feature = example.features.feature[feature_name]
    if feature.HasField('float_list'):
      f_v = '{0:.2f}'.format(feature.float_list.value[0])
      if f_v in stat:
        stat[f_v] += 1
      else:
        stat[f_v] = 1

  _write_stats(stat, output_file, len(examples))

def _analyze_sparse_feature(examples, feature_name, output_file):
  stat = {}
  for example in examples:
    feature = example.features.feature[feature_name]
    if feature.HasField('bytes_list'):
      f_v = feature.bytes_list.value[0].decode("utf-8")
      if f_v in stat:
        stat[f_v] += 1
      else:
        stat[f_v] = 1

  _write_stats(stat, output_file, len(examples))

def _analyze_nlp_feature(examples, feature_name, output_file):
  stat = {}
  for example in examples:
    feature = example.features.feature[feature_name]
    if feature.HasField('bytes_list'):
      f_v = ''
      for byte_list in feature.bytes_list.value:
        f_v += byte_list.decode('utf-8')
        f_v += ';'
      if f_v in stat:
        stat[f_v] += 1
      else:
        stat[f_v] = 1

  _write_stats(stat, output_file, len(examples))

def _analyze_examples(examples, suffix):
  for float_feature in float_features:
    output_file = output_dir + float_feature + suffix
    _analyze_float_feature(examples, float_feature, output_file)

  for sparse_feature in sparse_features:
    output_file = output_dir + sparse_feature + suffix
    _analyze_sparse_feature(examples, sparse_feature, output_file)

  for nlp_feature in nlp_features:
    output_file = output_dir + nlp_feature + suffix
    _analyze_nlp_feature(examples, nlp_feature, output_file)

def main(_):
  print('Started to analyze ....')
  train_examples = _read_tf_examples(train_filename)
  _analyze_examples(train_examples, '.train')
  eval_examples = _read_tf_examples(eval_filename)
  _analyze_examples(eval_examples, '.eval')
  test_examples = _read_tf_examples(test_filename)
  _analyze_examples(test_examples, '.test')
  print('Done!  Analysis finished!')


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
