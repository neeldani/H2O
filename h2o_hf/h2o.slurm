#!/bin/bash
#SBATCH --job-name=kivi
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bcjw-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/kivi_llama_7b.out
#SBATCH --mem=170G
#SBATCH --constraint=perf,nvperf

source ~/.bashrc
module load cuda/12.3.0

export HF_DATASETS_CACHE="/u/ndani/.cache"
export HF_HOME=$HF_DATASETS_CACHE
export HF_HUB_CACHE=$HF_DATASETS_CACHE
export HF_ASSETS_CACHE=$HF_DATASETS_CACHE
export TRANSFORMERS_CACHE=$HF_DATASETS_CACHE

export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/u/ndani/.conda/h2o/lib:$LD_LIBRARY_PATH

conda activate h2o

srun mem_spd_test.py