#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

REPO_PATH=$PROJECT_training2411/$USER/HDCRS-school-2024
FINETUNE_PATH=$REPO_PATH/HLS-finetuning/notebooks

mkdir $FINETUNE_PATH/datasets
mkdir $FINETUNE_PATH/configs
mkdir $FINETUNE_PATH/models

source "${ABSOLUTE_PATH}"/symlink.sh
source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

source "${ABSOLUTE_PATH}"/activate.sh

python3 -m pip install --upgrade -r "${ABSOLUTE_PATH}"/requirements.txt
mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html
pip install git+https://github.com/NASA-IMPACT/hls-foundation-os.git
