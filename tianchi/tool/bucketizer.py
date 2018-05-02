# This is a utility to generate bucket boundaries.
# Input is feature, training TF examples, and number of buckets;
# out put is the boundaries.
#
# Algorithm: sort the values and split into buckets so that each bucket
# contains same number of examples.

import tensorflow as tf

train_filename = '../data/v1/train.tfrecords'

def get_feature_values_(feature_name):
  values = []

  example_cnt = 0
  empty_feature_cnt = 0

  record_iterator = tf.python_io.tf_record_iterator(train_filename)
  for str_record in record_iterator:
    example_cnt += 1
    example = tf.train.Example()
    example.ParseFromString(str_record)
    feature = example.features.feature[feature_name]
    if len(feature.float_list.value) > 0:
      values.append(feature.float_list.value[0])
    else:
      empty_feature_cnt += 1

  print('\n\nAmong {} examples, {} do not have this feature.\n'.format(
    example_cnt, empty_feature_cnt))

  values.sort()
  return values

def main(argv):
  feature_name = argv[1]
  bucket_size = int(argv[2])

  values = get_feature_values_(feature_name)
  example_cnt = int(len(values) / bucket_size)

  print('Get {} values; {} examples in each bucket'.format(len(values), example_cnt))

  idx = example_cnt
  result = []
  while True:
    if idx >= len(values):
      break
    result.append('{:0.2f}'.format(values[idx]))
    idx += example_cnt

  print('\n\n***---***---*** Go ahead and copy me! :D :D ***---***---\n\n')
  print(', '.join(result))
  print('\n\n***---***---*** Mission completed! :D :D ***---***---\n\n')

if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
