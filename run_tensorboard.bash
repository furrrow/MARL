#!/bin/bash
source ~/mambaforge/etc/profile.d/conda.sh
conda activate rl
#tensorboard --logdir runs/simple_speaker_listener --port=8008
#tensorboard --logdir runs/balance --port=6006
tensorboard --logdir runs/simple --port=6006



