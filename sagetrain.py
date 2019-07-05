import os
import argparse
import sagemaker
from sagemaker import get_execution_role
from sagemaker.session import Session

from sagemaker.tensorflow import TensorFlow

import tensorflow as tf

if __name__ == '__main__':

    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = "ttam-sagemaker-test"#Session().default_bucket()

    # Location to save your custom code in tar.gz format.
    custom_code_upload_location = 's3://{}/customcode/tensorflow_iris'.format(bucket)

    # Location where results of model training are saved.
    model_artifacts_location = 's3://{}/artifacts'.format(bucket)

    # IAM execution role that gives SageMaker access to resources in your AWS account.
    role = get_execution_role()

    # Parser
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()

    sagemaker_session = sagemaker.Session()

    role = get_execution_role()

    # TODO get dataset ready

    estimator = TensorFlow(entry_point='unirep_estimator.py',
                           role=role,
                           framework_version='1.14.0',
                           training_steps=1000,
                           evaluation_steps=100,
                           train_instance_count=1,
                           train_instance_type='ml.p2.xlarge')

    # TODO figure out inputs
    estimator.fit('s3://{}/unirep'.format(bucket))