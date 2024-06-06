#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

REPO_PATH=$PROJECT_training2411/$USER/HDCRS-school-2024
FINETUNE_PATH=$REPO_PATH/HLS-finetuning/notebooks

module --force purge

ml Stages/2024
ml CUDA

source "${ABSOLUTE_PATH}"/symlink.sh
source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"
source "${ABSOLUTE_PATH}"/activate.sh

git clone https://github.com/meta-llama/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install -e .

source "${ABSOLUTE_PATH}"/create_kernel.sh
source "${ABSOLUTE_PATH}"/create_python_for_vscode.sh
