import functools
import logging
from pathlib import Path
import sys

import tensorflow as tf

# Setup logging
Path('results').mkdir(exist_ok=True)
#tf.compat.v1.logging.set_verbosity
tf.get_logger().setLevel(logging.INFO)
#tf.get_logger.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


# def generator_fn():
#     for digit in range(2):
#         line = 'I am digit {}'.format(digit)
#         words = line.split()
#         yield [w.encode() for w in words], len(words)
#
# for words in generator_fn():
#     print(words)

shapes = ([None], ())
types = (tf.string, tf.int32)

# dataset = tf.data.Dataset.from_generator(generator_fn, output_shapes=shapes, output_types=types)

#import tensorflow as tf
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.enable_eager_execution()

# for tf_words, tf_size in dataset:
#     print(tf_words, tf_size)


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
     params = params if params is not None else {}
     shapes = (([None], ()), [None])
     types = ((tf.string, tf.int32), tf.string)
     defaults = (('<pad>', 0), 'O')

     dataset = tf.data.Dataset.from_generator(
         functools.partial(generator_fn, words, tags),
         output_shapes=shapes, output_types=types)

     if shuffle_and_repeat:
         dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

     dataset = (dataset
                .padded_batch(params.get('batch_size', 1), shapes, defaults)
                .prefetch(1))
     return dataset


#tf.compat.v1.data.make_one_shot_iterator()
dataset = input_fn('/Users/sulabhkothari/PycharmProjects/ner/data/words.txt', '/Users/sulabhkothari/PycharmProjects/ner/data/tags.txt')
#iterator = dataset.enumerate(0)
#tf.compat.v1.data.Iterator.get_next()
for word,tag in dataset:
 print(word)
 print("==============================\n")



def model_fn(features, labels, mode, params):
    dropout = params['dropout']
    words, nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    tf.keras
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=1)
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Define the inference graph
    graph_outputs = some_tensorflow_applied_to(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Extract the predictions
        predictions = some_dict_from(graph_outputs)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Compute loss, metrics, tensorboard summaries
        loss = compute_loss_from(graph_outputs, labels)
        metrics = compute_metrics_from(graph_outputs, labels)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            # Get train operator
            train_op = compute_train_op_from(graph_outputs, labels)
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))
