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
import functools

import argparse
import pdb
import unirep

# Code from data_utils
# Lookup tables
aa_to_int = {
    'M': 1,
    'R': 2,
    'H': 3,
    'K': 4,
    'D': 5,
    'E': 6,
    'S': 7,
    'T': 8,
    'N': 9,
    'Q': 10,
    'C': 11,
    'U': 12,
    'G': 13,
    'P': 14,
    'A': 15,
    'V': 16,
    'I': 17,
    'F': 18,
    'Y': 19,
    'W': 20,
    'L': 21,
    'O': 22,  # Pyrrolysine
    'X': 23,  # Unknown
    'Z': 23,  # Glutamic acid or GLutamine
    'B': 23,  # Asparagine or aspartic acid
    'J': 23,  # Leucine or isoleucine
    'start': 24,
    'stop': 25,
}

int_to_aa = {value: key for key, value in aa_to_int.items()}


def get_aa_to_int():
    """
    Get the lookup table (for easy import)
    """
    return aa_to_int


def get_int_to_aa():
    """
    Get the lookup table (for easy import)
    """
    return int_to_aa


# Helper functions

def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]


def int_seq_to_aa(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return "".join([int_to_aa[i] for i in s])


def tf_str_len(s):
    """
    Returns length of tf.string s
    """
    return tf.size(tf.string_split([s], ""))


def tf_rank1_tensor_len(t):
    """
    Returns the length of a rank 1 tensor t as rank 0 int32
    """
    l = tf.reduce_sum(tf.sign(tf.abs(t)), 0)
    return tf.cast(l, tf.int32)


def tf_seq_to_tensor(s):
    """
    Input a tf.string of comma seperated integers.
    Returns Rank 1 tensor the length of the input sequence of type int32
    """
    return tf.string_to_number(
        tf.sparse_tensor_to_dense(tf.string_split([s], ","), default_value='0'), out_type=tf.int32
    )[0]


def smart_length(length, bucket_bounds=tf.constant([128, 256])):
    """
    Hash the given length into the windows given by bucket bounds.
    """
    # num_buckets = tf_len(bucket_bounds) + tf.constant(1)
    # Subtract length so that smaller bins are negative, then take sign
    # Eg: len is 129, sign = [-1,1]
    signed = tf.sign(bucket_bounds - length)

    # Now make 1 everywhere that length is greater than bound, else 0
    greater = tf.sign(tf.abs(signed - tf.constant(1)))

    # Now simply sum to count the number of bounds smaller than length
    key = tf.cast(tf.reduce_sum(greater), tf.int64)

    # This will be between 0 and len(bucket_bounds)
    return key


def pad_batch(ds, batch_size, padding=None, padded_shapes=([None])):
    """
    Helper for bucket batch pad- pads with zeros
    """
    return ds.padded_batch(batch_size,
                           padded_shapes=padded_shapes,
                           padding_values=padding
                           )


def aas_to_int_seq(aa_seq):
    int_seq = ""
    for aa in aa_seq:
        int_seq += str(aa_to_int[aa]) + ","
    return str(aa_to_int['start']) + "," + int_seq + str(aa_to_int['stop'])


# Preprocessing in python
def fasta_to_input_format(source, destination):
    # I don't know exactly how to do this in tf, so resorting to python.
    # Should go line by line so everything is not loaded into memory

    sourcefile = os.path.join(source)
    destination = os.path.join(destination)
    with open(sourcefile, 'r') as f:
        with open(destination, 'w') as dest:
            seq = ""
            for line in f:
                if line[0] == '>' and not seq == "":
                    dest.write(aas_to_int_seq(seq) + '\n')
                    seq = ""
                elif not line[0] == '>':
                    seq += line.replace("\n", "")


# Real data pipelines

def bucketbatchpad(
        batch_size=256,
        path_to_data=os.path.join("./data/SwissProt/sprot_ints.fasta"),  # Preprocessed- see note
        compressed="",  # See tf.contrib.data.TextLineDataset init args
        bounds=[128, 256],  # Default buckets of < 128, 128><256, >256
        # Unclear exactly what this does, should proly equal batchsize
        window_size=256,  # NOT a tensor
        padding=None,  # Use default padding of zero, otherwise see Dataset docs
        shuffle_buffer=None,  # None or the size of the buffer to shuffle with
        pad_shape=([None]),
        repeat=1,
        filt=None

):
    """
    Streams data from path_to_data that is correctly preprocessed.
    Divides into buckets given by bounds and pads to full length.
    Returns a dataset which will return a padded batch of batchsize
    with iteration.
    """
    batch_size = tf.constant(batch_size, tf.int64)
    bounds = tf.constant(bounds)
    window_size = tf.constant(window_size, tf.int64)

    path_to_data = os.path.join(path_to_data)
    # Parse strings to tensors
    dataset = tf.data.TextLineDataset(path_to_data).map(tf_seq_to_tensor)
    if filt is not None:
        dataset = dataset.filter(filt)

    if shuffle_buffer:
        # Stream elements uniformly randomly from a buffer
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Apply a repeat. Because this is after the shuffle, all elements of the dataset should be seen before repeat.
    # See https://stackoverflow.com/questions/44132307/tf-contrib-data-dataset-repeat-with-shuffle-notice-epoch-end-mixed-epochs
    dataset = dataset.repeat(count=repeat)
    # Apply grouping to bucket and pad
    grouped_dataset = dataset.apply(tf.data.experimental.group_by_window(
        key_func=lambda seq: smart_length(tf_rank1_tensor_len(seq), bucket_bounds=bounds),  # choose a bucket
        reduce_func=lambda key, ds: pad_batch(ds, batch_size, padding=padding, padded_shapes=pad_shape),
        # apply reduce funtion to pad
        window_size=window_size))

    return grouped_dataset


def shufflebatch(
        batch_size=256,
        shuffle_buffer=None,
        repeat=1,
        path_to_data="./data/SwissProt/sprot_ints.fasta"
):
    """
    Draws from an (optionally shuffled) dataset, repeats dataset repeat times,
    and serves batches of the specified size.
    """

    path_to_data = os.path.join(path_to_data)
    # Parse strings to tensors
    dataset = tf.contrib.data.TextLineDataset(path_to_data).map(tf_seq_to_tensor)
    if shuffle_buffer:
        # Stream elements uniformly randomly from a buffer
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Apply a repeat. Because this is after the shuffle, all elements of the dataset should be seen before repeat.
    # See https://stackoverflow.com/questions/44132307/tf-contrib-data-dataset-repeat-with-shuffle-notice-epoch-end-mixed-epochs
    dataset = dataset.repeat(count=repeat)
    dataset = dataset.batch(batch_size)
    return dataset


# Functions from babbler1900 being moved to allow use of estimators

def is_valid_seq(seq, max_len=2000):
    """
    True if seq is valid for the babbler, False otherwise.
    """
    l = len(seq)
    valid_aas = "MRHKDESTNQCUGPAVIFYWLO"
    if (l < max_len) and set(seq) <= set(valid_aas):
        return True
    else:
        return False


def format_seq(seq, stop=False):
    """
    Takes an amino acid sequence, returns a list of integers in the codex of the babbler.
    Here, the default is to strip the stop symbol (stop=False) which would have
    otherwise been added to the end of the sequence. If you are trying to generate
    a rep, do not include the stop. It is probably best to ignore the stop if you are
    co-tuning the babbler and a top model as well.
    """
    if stop:
        int_seq = aa_seq_to_int(seq.strip())
    else:
        int_seq = aa_seq_to_int(seq.strip())[:-1]
    return int_seq


def bucket_batch_pad(filepath, batch_size=12, shuffle_buffer=10000, upper=2000, lower=50, interval=10):
    """
    Read sequences from a filepath, batch them into buckets of similar lengths, and
    pad out to the longest sequence.
    Upper, lower and interval define how the buckets are created.
    Any sequence shorter than lower will be grouped together, as with any greater
    than upper. Interval defines the "walls" of all the other buckets.
    WARNING: Define large intervals for small datasets because the default behavior
    is to repeat the same sequence to fill a batch. If there is only one sequence
    within a bucket, it will be repeated batch_size -1 times to fill the batch.
    """

    bucket = [lower + (i * interval) for i in range(int(upper / interval))]
    bucket_batch = bucketbatchpad(
        batch_size=batch_size,
        pad_shape=([None]),
        window_size=batch_size,
        bounds=bucket,
        path_to_data=filepath,
        shuffle_buffer=shuffle_buffer,
        repeat=None
    )  # .make_one_shot_iterator().get_next()
    return bucket_batch


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
                    if is_valid_seq(seq) and len(seq) < 275:
                        formatted = ",".join(map(str, format_seq(seq)))
                        destination.write(formatted)
                        destination.write('\n')

        # dataset = tf.data.Dataset.from_tensor_slices(store)

        bucket_op = bucket_batch_pad("formatted.txt", 12, interval=1000)

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

    y = features
    #y = tf.fill([1], 42 * batch_size)
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
        tf.cast(pad_adjusted_targets, tf.int32),
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

    dataset = bucket_batch_pad("formatted.txt", 12, interval=1000)
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
                                       model_dir="results/",
                                       params={
                                           'batch_size': 12,
                                           'model_path': "./1900_weights",
                                           'lr': 0.001
                                       })

    #estimator.train(input_fn())

    #hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    #input_fun = functools.partial(input_gen, 12)
    input_fun = functools.partial(bucket_batch_pad,"formatted.txt", 12, interval=1000)
    train_spec = tf.estimator.TrainSpec(input_fn=input_fun)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fun, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    #pdb.set_trace()
