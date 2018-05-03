# Built a multi-head DNN model.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from google.protobuf import text_format

import feature as f
import hparams as hp
import tensorflow as tf
import time

train_filename = '../data/v1/train.tfrecords'
eval_filename  = '../data/v1/eval.tfrecords'
test_filename  = '../data/v1/test.tfrecords'

model_dir = 'model/'


def main(argv):
  """Builds, trains, and evaluates the model."""
  eval_only = False
  if len(argv) > 1:
    eval_only = (argv[1] == 'eval')

  def input_train(eval = False):
    data_files = eval_filename if eval else train_filename
    feature_columns = f.build_feature_columns()
    feature_spec = tf.estimator.classifier_parse_example_spec(
        feature_columns, label_key=hp.label_key_, label_dtype=tf.float32)
    dataset = tf.contrib.data.make_batched_features_dataset(
        data_files,
        hp.batch_size,
        feature_spec,
        num_epochs=1 if eval else None,
    )

    batch_features = dataset.make_one_shot_iterator().get_next()
    label = batch_features.pop(hp.label_key_)
    return batch_features, label

  def input_pred():
    feature_columns = f.build_feature_columns()
    feature_spec = tf.estimator.classifier_parse_example_spec(
        feature_columns, label_key=hp.label_key_, label_dtype=tf.float32)
    dataset = tf.contrib.data.make_batched_features_dataset(
        test_filename,
        hp.batch_size,
        feature_spec,
        shuffle=False,
        num_epochs=1,
    )
    batch_features = dataset.make_one_shot_iterator().get_next()
    return batch_features

  model_config = tf.estimator.RunConfig(
    keep_checkpoint_max=3,
    save_checkpoints_steps=hp.save_checkpoints_steps,
    model_dir=model_dir + hp.label_key_ + hp.model_version)

  model = tf.estimator.DNNRegressor(
    hidden_units=hp.hidden_units,
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=hp.learning_rate,
      l1_regularization_strength=hp.l1_regularization_strength),
    feature_columns=f.build_feature_columns(),
    label_dimension=1,
    dropout=0.5,
    config=model_config)

  train_spec = tf.estimator.TrainSpec(input_fn=input_train, max_steps=hp.train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_train(True))

  if eval_only:
    for _ in range(120):
      time.sleep(3)
      model.evaluate(input_fn=lambda: input_train(True))
  else:
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
        del feature[hp.label_key_].bytes_list.value[:]
        feature[hp.label_key_].float_list.value.append(scores[idx])
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
