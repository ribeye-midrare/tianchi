# It reads from the tensorflow proto buffer and convert it to CSV file
# Input is tfrecords file; fields are given in a string separated by comma

import csv
import tensorflow as tf

tfrecord_file = '../data/v1/test.tfrecords'
csv_file       = '../data/v2/submission.csv'
fields_to_inc  = 'vid,shousuoya,shuzhangya,ganyousanzhi,gaomiduzhidanbai,dimiduzhidanbai'

def get_feature_value(example, key):
  feature = example.features.feature[key]
  if feature.HasField('bytes_list'):
    return feature.bytes_list.value[0].decode("utf-8")
  elif feature.HasField('float_list'):
    return '{0:.3f}'.format(feature.float_list.value[0])
  elif feature.HasField('int64_list'):
    return str(feature.int64_list.value[0])
  else:
    raise TypeError('Feature %s has unexpected field type' % key)

def convert_to_str(example):
  """Converts from tfExample to CSV str, end with \n."""
  fields = fields_to_inc.split(",");
  feature = example.features.feature
  str_result = ''
  is_first = True
  for feature in fields:
    if is_first:
      is_first = False
    else:
      str_result += ','
    str_result += get_feature_value(example, feature)

  return str_result + '\n'

def main(_):
  record_iterator = tf.python_io.tf_record_iterator(tfrecord_file)
  with open(csv_file, 'w+') as f:
    for str_record in record_iterator:
      example = tf.train.Example()
      example.ParseFromString(str_record)
      f.write(convert_to_str(example))

if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
