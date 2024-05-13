#!/usr/bin/env bash
# Slurm job configuration
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --account=training2411
#SBATCH --output=output.out
#SBATCH --error=error.er
#SBATCH --time=2:00:00
#SBATCH --job-name=<identifier>
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
#SBATCH --gpus-per-node=4
#SBATCH --hint=nomultithread

set -x

PARTITION=training2411
JOB_NAME=<identifier>
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

module --force purge

ml Stages/2023
ml CUDA/11.7

source /p/project/training2411/<user>/miniconda/bin/activate py39

echo "Starting new training"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=8 \
    --nodes=2 \
    --ntasks-per-node=4 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train.py <config file path> --launcher="slurm" ${PY_ARGS}