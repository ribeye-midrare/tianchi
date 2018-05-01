# Reads some proto examples from the given file.  For debugging purpose.

from google.protobuf import text_format
import tensorflow as tf

def main(argv):
  """The first argument is TF records file path.
     Second argument is the example count to read.
  """
  assert len(argv) == 3
  filename = argv[1]
  record_count = int(argv[2])

  record_iterator = tf.python_io.tf_record_iterator(filename)
  for _ in range(record_count):
    example = tf.train.Example()
    example.ParseFromString(next(record_iterator))
    print(text_format.MessageToString(example, as_utf8=True))
    print('\n')


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
