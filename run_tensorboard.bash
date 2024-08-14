#!/bin/bash
source ~/mambaforge/etc/profile.d/conda.sh
conda activate rl
#tensorboard --logdir runs/simple_speaker_listener
tensorboard --logdir runs/balance



