import tensorflow as tf
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from data_utils import aa_seq_to_int, int_to_aa, bucketbatchpad
import os
import unirep

import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def model_fn(features, labels, mode, params):

    model_path = os.path.join()
    batch_size = 12
    model = unirep.babbler1900(batch_size=batch_size, model_path=model_path)