import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

import tensorflow as tf

sagemaker_session = sagemaker.Session()

role = get_execution_role()

# TODO get dataset ready

estimator = TensorFlow(entry_point='unirep_estimator.py',
                       role=role,
                       framework_version='1.12.0',
                       training_steps=1000,
                       evaluation_steps=100,
                       train_instance_count=1,
                       train_instance_type='ml.p2.xlarge')

# TODO figure out inputs
estimator.fit()