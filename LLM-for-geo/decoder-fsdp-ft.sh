#!/usr/bin/env bash
# Slurm job configuration
#SBATCH --account=training2411
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gpus-per-node=4
#SBATCH --mem=256gb
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output.out
#SBATCH --error=error.er
#SBATCH --time=2:00:00
#SBATCH --job-name=peft
#SBATCH --partition=dc-gpu-devel
#SBATCH --hint=nomultithread

set -x

# PARTITION=dc-gpu-devel
JOB_NAME=peft
# SRUN_ARGS=${SRUN_ARGS:-""}
# PY_ARGS=${@:4}

module --force purge

ml Stages/2024
ml CUDA

source /p/project/training2411/kumar/HDCRS-school-2024/.venv/bin/activate

echo "Starting LLM peftraining"

export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
export SRUN_CPUS_PER_TASK=12
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR}i" # Assuming you want to append 'i' to the hostname.
export MASTER_ADDR
export MASTER_PORT=5557
export GPUS_PER_NODE=4

export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER="efa"
export NCCL_TREE_THRESHOLD=0

export NCCL_DEBUG=INFO
export NCCL_SOCKET_TIMEOUT=600000 # Set the timeout to 10 minutes (60000 milliseconds)
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=OFF

export NCCL_P2P_DISABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"
export LOGLEVEL=INFO

export NCCL_DEBUG_SUBSYS=INFO
export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/p/project/training2411/ramasubramanian1/HDCRS-school-2024/sc_venv_template/venv/lib/

# echo "Using torch from $(python -c 'import torch; print(torch.__file__)')"
# echo "Using torch cuda from $(python -c 'import torch; print(torch.version.cuda)')"
# echo "Using nccl from $(python -c 'import torch; print(torch.cuda.nccl.version())')"


GLOBAL_BATCH_SIZE=64
MAX_BATCH_SIZE=8
GRAD_ACCUM_STEPS=1

srun torchrun --nnodes 1 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint localhost:0 llama-recipes/recipes/finetuning/finetuning.py --model_name /p/project/training2411/kumar/meta-llama/Llama-2-7b-chat-hf --dataset "custom_dataset" --custom_dataset.file "/p/project/training2411/kumar/HDCRS-school-2024/LLM-for-geo/custom_dataset.py" --enable_fsdp --use_peft --peft_method lora --output_dir output_dir --batching-strategy padding
