# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pathlib
import json

import experiment


def get_args():
    """Define the task arguments with the default values.

    Returns:
        experiment parameters
    """
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/AntBulletEnv-v0.yaml')
    p.add_argument(
        '--task-name',
        help='The name of the environment to use',
        required=True)

    p.add_argument(
        '--alg',
        help='The algorithm to be used for training',
        choices=["A2C", "PPO"],
        required=True)

    p.add_argument(
        '--policy',
        help='The type of model to use.',
        choices=["GnnPolicy", "MlpPolicy"],
        required=True)

    p.add_argument(
        "--total-timesteps",
        help="The total number of samples (env steps) to train on",
        type=int,
        default=10000
    )

    p.add_argument(
        '--tensorboard-log',
        help='the log location for tensorboard (if None, no logging)',
        default="log")

    # Algorithm (PPO/A2C) arguments
    p.add_argument(
        '--n-steps',
        help='The number of steps to run for each environment per update',
        type=int)

    p.add_argument(
        '--n-epochs',
        help="For PPO: Number of epochs when optimizing the surrogate loss.",
        type=int,
    )
    p.add_argument(
        '--seed',
        help='Random seed (default: 42)',
        type=int,
    )
    p.add_argument(
        '--device',
        help='Device (cpu, cuda, ...) on which the code should be run.'
             'Setting it to auto, the code will be run on the GPU if possible.',
        default="auto")

    # Policy arguments
    p.add_argument(
        '--net_arch',
        help='The specification of the policy and value networks',
        type=json.loads)

    p.add_argument(
        '--activation-fn',
        help='Activation function of the policy and value networks',
        choices=["Tanh", "ReLU"])

    # Estimator arguments
    p.add_argument(
        '--learning-rate',
        help='Learning rate value for the optimizers.',
        type=float)

    # Saved model arguments
    p.add_argument(
        '--job-dir',
        help='GCS location to export models')
    p.add_argument(
        '--model-name',
        help='The name of your saved model',
        default='model.pth.zip')

    return p.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    experiment.run(args)


if __name__ == '__main__':
    main()
