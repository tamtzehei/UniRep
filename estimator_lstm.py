import tensorflow as tf
# tf.enable_eager_execution()

import logging
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import numpy as np
import pandas as pd
import sys
from data_utils import aa_seq_to_int, int_to_aa, bucketbatchpad
import os
import unirep
import data_utils
import functools

import argparse
import pdb
import unirep



# Helpers
def tf_get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def sample_with_temp(logits, t):
    """
    Takes temperature between 0 and 1 -> zero most conservative, 1 most liberal. Samples.
    """
    t_adjusted = logits / t  # broadcast temperature normalization
    softed = tf.nn.softmax(t_adjusted)

    # Make a categorical distribution from the softmax and sample
    return tf.distributions.Categorical(probs=softed).sample()

def nonpad_len(batch):
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=0)
    return lengths

def input_gen(batch_size):

    while True:

        with open("seqs.txt", "r") as source:
            with open("formatted.txt", "w") as destination:
                for i, seq in enumerate(source):
                    seq = seq.strip()
                    if data_utils.is_valid_seq(seq) and len(seq) < 275:
                        formatted = ",".join(map(str, data_utils.format_seq(seq)))
                        destination.write(formatted)
                        destination.write('\n')

        # dataset = tf.data.Dataset.from_tensor_slices(store)

        bucket_op = data_utils.bucket_batch_pad("formatted.txt", 12, interval=1000)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch = sess.run(bucket_op)

        print batch
        print batch.shape
        print type(batch)

        # Replication of feed_dict from unirep_tutorial.py
        # batch is np.ndarray
        x = batch
        y = [[42]] * batch_size
        # batch_size
        seq_length = nonpad_len(batch)
        #initial_state, _ = rnn.zero_state(12, tf.float32)

        #print type(initial_state)
        #print initial_state.shape

        features = batch


        labels = None
        """
        return tf.estimator.inputs.numpy_input_fn(
            'x': tf.convert_to_tensor(x),
            'y': tf.convert_to_tensor(y),
            'batch_size': tf.convert_to_tensor(batch_size),
             'seq_length': tf.convert_to_tensor(seq_length),
             'initial_state': tf.convert_to_tensor(initial_state)
        )
        """
        return features#, labels

def input_fn():
    shapes = ([12, 265])
    types = (tf.int16)

    dataset = tf.data.Dataset.from_generator(functools.partial(input_gen, 12), output_types=types)

    for tens in dataset:
        print tens

    labels = None

    return dataset


def model_fn(features, labels, mode, params):
    # Define the inference graph

    #length = tf.py_func(nonpad_len, [features], stateful=False)
    length = tf.py_func(nonpad_len, [features], [tf.int32])[0]

    rnn_size = 1900
    vocab_size = 26
    model_path = params['model_path']
    batch_size = params['batch_size']
    learning_rate = params['lr']

    y = tf.fill([1], 42 * batch_size)
    rnn = unirep.mLSTMCell1900(rnn_size,
                    model_path=model_path,
                        wn=True)
    zero_state = rnn.zero_state(batch_size, tf.float32)
    single_zero = rnn.zero_state(1, tf.float32)
    mask = tf.sign(y)  # 1 for nonpad, zero for pad
    inverse_mask = 1 - mask  # 0 for nonpad, 1 for pad

    total_padded = tf.reduce_sum(inverse_mask)

    pad_adjusted_targets = (y - 1) + inverse_mask

    embed_matrix = tf.get_variable(
        "embed_matrix", dtype=tf.float32, initializer=np.load(os.path.join(model_path, "embed_matrix:0.npy"))
    )
    embed_cell = tf.nn.embedding_lookup(embed_matrix, features)

    # In progress
    initial_state = (tf.zeros([batch_size, rnn_size], dtype=tf.float32), tf.zeros([batch_size, rnn_size], dtype=tf.float32))

    output, final_state = tf.nn.dynamic_rnn(
        rnn,
        embed_cell,
        initial_state=initial_state,
        swap_memory=True,
        parallel_iterations=1
    )

    indices = length - 1
    top_final_hidden = tf.gather_nd(output,
                                          tf.stack([tf.range(tf_get_shape(output)[0], dtype=tf.int32), indices],
                                                   axis=1))
    flat = tf.reshape(output, [-1, rnn_size])
    logits_flat = tf.contrib.layers.fully_connected(
        flat, vocab_size - 1, activation_fn=None,
        weights_initializer=tf.constant_initializer(
            np.load(os.path.join(model_path, "fully_connected_weights:0.npy"))),
        biases_initializer=tf.constant_initializer(
            np.load(os.path.join(model_path, "fully_connected_biases:0.npy"))))
    logits = tf.reshape(
        logits_flat, [batch_size, tf_get_shape(features)[1], vocab_size - 1])

    # Loss
    batch_losses = tf.contrib.seq2seq.sequence_loss(
        logits,
        tf.cast(features, tf.int32),
        #tf.cast(pad_adjusted_targets, tf.int32),
        tf.cast(mask, tf.float32),
        average_across_batch=False
    )
    loss = tf.reduce_mean(batch_losses)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Equivalent to all_step_op
    train_op = optimizer.minimize(loss)#, global_step=tf.train.get_or_create_global_step())


    if mode == tf.estimator.ModeKeys.PREDICT:
        # Extract the predictions
        predictions = logits
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        # Compute loss, metrics, tensorboard summaries
        #loss = compute_loss_from(graph_outputs, labels)
        #metrics = compute_metrics_from(graph_outputs, labels)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            # Get train operator

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)

        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))

if __name__ == '__main__':

    # Test data input

    dataset = data_utils.bucket_batch_pad("formatted.txt", 12, interval=1000)
    iterator = dataset.make_one_shot_iterator()
    node = iterator.get_next()
    with tf.Session() as sess:
        print(sess.run(node))


    # Build feature columns
    """
    feature_columns = []
    feature_columns.append(tf.feature_column.numeric_column(key="x", shape = [12, 265]))
    feature_columns.append(tf.feature_column.numeric_column(key="y"))
    feature_columns.append(tf.feature_column.numeric_column(key="batch_size"))
    feature_columns.append(tf.feature_column.numeric_column(key="seq_length"))
    """



    # Test estimator
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('results/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params={
                                           'batch_size': 12,
                                           'model_path': "./1900_weights",
                                           'lr': 0.001
                                       })

    #estimator.train(input_fn())

    #hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    #input_fun = functools.partial(input_gen, 12)
    input_fun = functools.partial(data_utils.bucket_batch_pad,"formatted.txt", 12, interval=1000)
    train_spec = tf.estimator.TrainSpec(input_fn=input_fun)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fun, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    #pdb.set_trace()


