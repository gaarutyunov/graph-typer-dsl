#!/usr/bin/env bash

# install requirements
pip install torch==1.11.0 torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.11.0)
pip install lmdb
pip install tensorboardX==2.4.1
pip install performer-pytorch
pip install tensorboard
pip install setuptools==59.5.0
pip install dpu-utils
pip install fairseq

