# Covert the input CSV files into tfrecords file.
# V1 uses only float and sparse features

from random import shuffle
import codecs
import csv
import tensorflow as tf

train_filename = '../data/v1/train.tfrecords'
eval_filename  = '../data/v1/eval.tfrecords'
test_filename  = '../data/v1/test.tfrecords'
feature_dir    = '../data/feature_set/'

raw_feature_file1 = '../data/data_part1_20180408.txt'
raw_feature_file2 = '../data/data_part2_20180408.txt'
raw_train_file    = '../data/train_20180408.csv'  # 38199
raw_test_filename = '../data/test_a_20180409.csv' # 9538


float_features  = {'1814', '1815', '1850', '190', '191', '192', '2403', '2404', '2405', '10004'}
sparse_features = {'2302', '1840', '3190', '3191', '3192', '3193', '3195', '3196', '3197',
                   '3429', '3430', '3730', '300005', '0407', '0420', '0421', '0423', '0426', '0430',
                   '0431', '0901', '100010'}


def _string_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, encoding='utf-8')]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def build_example(vid, features, labels):
  # create the tfExample proto with features and labels
  # .strip()
  feature = {}
  for feature_name in float_features:
    if feature_name in features:
      stripped_value = features[feature_name].strip()
      if stripped_value:
        try:
          f_value = float(stripped_value.replace(' ', ''))
          feature[feature_name] = _float_feature(f_value)
        except ValueError:
          print('Float feature {} contains non-float value {}'.format(feature_name, stripped_value))

  for feature_name in sparse_features:
    if feature_name in features:
      stripped_value = features[feature_name].strip()
      if stripped_value:
        feature[feature_name] = _string_feature(stripped_value)

  for label, value in labels.items():
    if label != 'vid':
      if not value:
        feature[label] = _float_feature(0.0)
      else:
        try:
          f_value = float(value.strip())
          feature[label] = _float_feature(f_value)
        except ValueError:
          # do not allow bad labels
          print('label {} has non-float value {}'.format(label, value))
          return None

  feature['vid'] = _string_feature(vid)
  return tf.train.Example(features=tf.train.Features(feature=feature))

def read_label(file_name):
  # dict key: vid; dict value: dict of values
  labels = {}
  with open(file_name, newline='', encoding='gb18030') as label_file:
    label_reader = csv.DictReader(label_file)  # default delimiter is ','
    for row in label_reader:
      labels[row['vid']] = row

  print(file_name, ' contains label count: ', len(labels))
  return labels


def read_feature():
  features = {}
  with open(raw_feature_file1, newline='') as feature_file:
    feature_reader = csv.DictReader(feature_file, delimiter='$')
    for row in feature_reader:
      if row['\ufeffvid'] in features:
        features[row['\ufeffvid']][row['table_id']] = row['field_results']
      else:
        features[row['\ufeffvid']] = {row['table_id']: row['field_results']}

  with open(raw_feature_file2, newline='') as feature_file:
    feature_reader = csv.DictReader(feature_file, delimiter='$')
    for row in feature_reader:
      if row['\ufeffvid'] in features:
        features[row['\ufeffvid']][row['table_id']] = row['field_results']
      else:
        features[row['\ufeffvid']] = {row['table_id']: row['field_results']}

  print("Data file contains key count: ", len(features))
  return features


def print_all_features(features):
  feature_set = {}
  for vid, value in features.items():
    for feature_name, feature_value in value.items():
      if feature_value.strip():
        if feature_name in feature_set:
          feature_set[feature_name]['value'].add(feature_value.strip())
          feature_set[feature_name]['count'] += 1
        else:
          feature_set[feature_name] = {'count': 1, 'value': {feature_value.strip()}}

  # # write to local files
  for feature_name, data in feature_set.items():
    filename = feature_dir + feature_name
    file = open(filename, 'w+')
    for v in data['value']:
      file.write('{}\n'.format(v))

    file.close()

  # print some statistics
  stats = open(feature_dir + 'stats.txt', 'w+')
  stats.write('unique feature count: {}\n'.format(len(feature_set)))
  stats.write('feature_name, unique_value, total_count\n')
  all_feature = []
  for feature_name, data in feature_set.items():
    all_feature.append({
      'name': feature_name,
      'unique': len(data['value']),
      'count': data['count']})


  for ele in sorted(all_feature, key = lambda i: i['count'], reverse=True):
    stats.write('{}, {}, {}\n'.format(ele['name'], ele['unique'], ele['count']))

  stats.close();


def convert_to_tfrecord(shuffling, train_labels, test_labels, features):
  test_size = 0
  print('----start to write test tf example------')
  test_example_file = tf.python_io.TFRecordWriter(test_filename)
  for vid, empty_label in test_labels.items():
    test_size += 1
    new_example = build_example(vid, features[vid], empty_label)
    test_example_file.write(new_example.SerializeToString())
  test_example_file.close()
  print('-----wrote test examples. size: ', test_size)

  training_size = 0
  eval_size = 0
  all_train_examples = []
  for vid, label in train_labels.items():
    example = build_example(vid, features[vid], label)
    if example:
      all_train_examples.append(example)
  if shuffling:
    shuffle(all_train_examples)

  training_size = int(0.8 * len(all_train_examples))
  eval_size = len(all_train_examples) - training_size

  print('-----start to write training/eval examples---------')
  train_example_file = tf.python_io.TFRecordWriter(train_filename)
  for example in all_train_examples[:training_size]:
    train_example_file.write(example.SerializeToString())
  train_example_file.close()

  eval_example_file  = tf.python_io.TFRecordWriter(eval_filename)
  for example in all_train_examples[-eval_size:]:
    eval_example_file.write(example.SerializeToString())
  eval_example_file.close()

  print('---wrote {} train examples and {} eval examples'.format(training_size, eval_size))

def main():
  # combine data part 1 and part 2 into a single feature file,
  # and read rhe train labels.  Then connect the feature and
  # label file to create a tf example proto
  # split the training into a 80/20 data set after shuffling
  train_labels = read_label(raw_train_file)
  test_labels = read_label(raw_test_filename)
  features = read_feature()
  print_all_features(features)
  convert_to_tfrecord(True, train_labels, test_labels, features)

if __name__ == "__main__": main()
