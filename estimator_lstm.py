import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import numpy as np
import pandas as pd
import sys
from data_utils import aa_seq_to_int, int_to_aa, bucketbatchpad
import os
import unirep
import data_utils

import argparse
import pdb
import unirep

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

def model_fn(features, labels, mode, params):
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




