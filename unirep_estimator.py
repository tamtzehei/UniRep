#!/home/tzehei/anaconda3/bin/python

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from data_utils import aa_seq_to_int, int_to_aa, bucketbatchpad
import os
import unirep
import data_utils

import argparse
import pdb
import unirep
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

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
    lengths = np.sum(nonzero, axis=1)
    return lengths

def model_fn(features, labels, mode, params):
    # Get pretrained weights
    # os.system('aws s3 sync --no-sign-request --quiet s3://unirep-public/1900_weights/ 1900_weights/')

    model_path = "./1900_weights"

    batch_size = params['batch_size']

    # Define model


    model = unirep.babbler1900(batch_size=batch_size, model_path=model_path)

    bucket_op = model.bucket_batch_pad("formatted.txt", interval=1000)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch = sess.run(bucket_op)

    # Predict
    if mode == Modes.PREDICT:
        predictions = None
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Train and Evaluate
    final_hidden, x_placeholder, batch_size_placeholder, seq_length_placeholder, initial_state_placeholder = (
        model.get_rep_ops())

    y_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name="y")
    initializer = tf.contrib.layers.xavier_initializer(uniform=False)

    with tf.variable_scope("top"):
        prediction = tf.contrib.layers.fully_connected(
            final_hidden, 1, activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=tf.zeros_initializer()
        )

    loss = tf.losses.mean_squared_error(y_placeholder, prediction)
    learning_rate = .001
    top_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="top")
    optimizer = tf.train.AdamOptimizer(learning_rate)
    top_only_step_op = optimizer.minimize(loss, var_list=top_variables)
    all_step_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step)

    nonpad_len(batch)

    y = [[42]] * batch_size
    num_iters = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iters):
            batch = sess.run(bucket_op)
            length = nonpad_len(batch)
            loss_, __, = sess.run([loss, all_step_op],
                                  feed_dict={
                                      x_placeholder: batch,
                                      y_placeholder: y,
                                      batch_size_placeholder: batch_size,
                                      seq_length_placeholder: length,
                                      initial_state_placeholder: model._zero_state
                                  }
                                  )

            print("Iteration {0}: {1}".format(i, loss_))

    if mode == Modes.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss
        )

    elif mode == Modes.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=all_step_op
        )


# This currently only returns seqs, change to actual labels if using labeled data
def input_fn():
    store = np.array([])
    with open("seqs.txt", "r") as source:
        with open("formatted.txt", "w") as destination:
            for i, seq in enumerate(source):
                seq = seq.strip()
                if data_utils.is_valid_seq(seq) and len(seq) < 275:
                    formatted = ",".join(map(str, data_utils.format_seq(seq)))
                    destination.write(formatted)
                    destination.write('\n')
                    store = np.append(store, formatted)

    # dataset = tf.data.Dataset.from_tensor_slices(store)

    bucket_op = data_utils.bucket_batch_pad("formatted.txt", 12, interval=1000)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch = sess.run(bucket_op)

    features = batch
    labels = None

    return features, labels



# test code

estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   params={
                                       'batch_size': 12
                                   })

hook = tf.contrib.estimator.stop_if_no_increase_hook(
    estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
train_spec = tf.estimator.TrainSpec(input_fn=input_fn, hooks=[hook])
#eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
pdb.set_trace()
estimator.train(input_fn(), steps=1000)

