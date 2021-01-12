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

# Dockerfile-gpu
FROM nvidia/cuda:11.1-cudnn8-runtime

# Installs necessary dependencies.
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update

RUN apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN  python3 --version
RUN  pip3 --version

RUN pip3 install --upgrade pip

WORKDIR /root

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip3 install cloudml-hypertune

# Installs pandas, and google-cloud-storage.
RUN pip3 install pandas google-cloud-storage

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


# Get the tum-adlr-ws21-04 source code
# Option 1: clone the repo
#RUN git clone https://github.com/HannesStark/tum-adlr-ws21-04.git

# Option 2: copy the source code from local machine to the container
# this assumes you call docker build from the root of the tum-adlr-ws21-04 repo
COPY . .


# install dependencies for ADLR
RUN pip3 install -e .

# Installs pytorch and its libraries.
RUN pip3 install torch===1.7.0+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip3 install torch-geometric

# Install pybullet
RUN pip3 install pybullet --upgrade --user

# install stable_baselines3
RUN pip3 install -e git+https://github.com/DLR-RM/stable-baselines3#egg=stable-baselines3

# we need to install tensorflow as well so that pytorch can use the correct tensorboard version
# which allows to log tensorboard summaries to the google cloud storage (gs://example_bucket/example_file)
RUN pip3 install tensorflow

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "-u", "trainer/task.py"]