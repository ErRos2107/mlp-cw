#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o sample_experiment_outfile  # send stdout to sample_experiment_outfile
#SBATCH -e sample_experiment_errfile  # send stderr to sample_experiment_errfile
#SBATCH -t 7:59:59  # time requested in hour:minute:secon
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

python twitter_starter.py --typec rnn --experiment_prefix 1a2d3a4c5a6a --seed 25012018 --cell gru --hidden_unit 256 --dropout_rate 0 --num_units 1024 --batch_size 128 --epochs 50
