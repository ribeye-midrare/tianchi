# Built a multi-head DNN model.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from google.protobuf import text_format

import feature as f
import tensorflow as tf

train_filename = '../data/v1/train.tfrecords'
eval_filename  = '../data/v1/eval.tfrecords'
test_filename  = '../data/v1/test.tfrecords'

model_dir = 'model/v1/'
train_steps = 60000 # 60k steps
batch_size = 128
label_key_ = 'ganyousanzhi'


def main(argv):
  """Builds, trains, and evaluates the model."""

  def input_train(eval = False):
    data_files = eval_filename if eval else train_filename
    feature_columns = f.build_feature_columns()
    feature_spec = tf.estimator.classifier_parse_example_spec(
        feature_columns, label_key=label_key_, label_dtype=tf.float32)
    dataset = tf.contrib.data.make_batched_features_dataset(
        data_files,
        batch_size,
        feature_spec,
    )

    batch_features = dataset.make_one_shot_iterator().get_next()
    label = batch_features.pop(label_key_)
    return batch_features, label

  def input_pred():
    feature_columns = f.build_feature_columns()
    feature_spec = tf.estimator.classifier_parse_example_spec(
        feature_columns, label_key=label_key_, label_dtype=tf.float32)
    dataset = tf.contrib.data.make_batched_features_dataset(
        test_filename,
        batch_size,
        feature_spec,
        shuffle=False,
        num_epochs=1,
    )
    batch_features = dataset.make_one_shot_iterator().get_next()
    return batch_features

  model = tf.estimator.DNNRegressor(
    hidden_units=[512, 128, 128],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.01,
      l1_regularization_strength=0.001),
    feature_columns=f.build_feature_columns(),
    model_dir=model_dir + label_key_,
    label_dimension=1,
    dropout=0.5)

  train_spec = tf.estimator.TrainSpec(input_fn=input_train, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_train(True))

  tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

  predicted_output = model.predict(input_fn=input_pred)
  scores = []
  for pred_score in predicted_output:
    scores.append(pred_score['predictions'])
  print('Predict result length: {}'.format(len(scores)))

  # output to tfexamples
  output = []
  record_iterator = tf.python_io.tf_record_iterator(test_filename)
  for idx, str_example in enumerate(record_iterator):
    if idx >= len(scores):
      print('XXXXXXX bug! more examples than scores! XXXXXXX')
      return
    else:
      example = tf.train.Example()
      example.ParseFromString(str_example)
      feature = example.features.feature
      del feature[label_key_].bytes_list.value[:]
      feature[label_key_].float_list.value.append(scores[idx])
      output.append(example)

  print('-----start to write result examples---------')
  result_file = tf.python_io.TFRecordWriter(test_filename)
  for example in output:
    result_file.write(example.SerializeToString())
  result_file.close()
  print('-----wrote {} results to {}---------'.format(len(output), test_filename))

if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
