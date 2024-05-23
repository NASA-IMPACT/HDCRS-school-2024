#!/usr/bin/env bash
# Slurm job configuration
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --account=training2411
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --time=2:00:00
#SBATCH --job-name=<identifier>
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --gpus-per-node=4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=8

PARTITION=training2411
JOB_NAME=${USER}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

source sc_venv_template/activate.sh

# Without this, srun does not inherit cpus-per-task from sbatch.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# so processes know who to talk to
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010
export GPUS_PER_NODE=4


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
    python -u train.py configs/burn_scars_Prithvi_100M.py --launcher="slurm" ${PY_ARGS}
