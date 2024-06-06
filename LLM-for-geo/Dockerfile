# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

LABEL maintainer="NASA IMPACT"

RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    apt-get install -y nginx && \
    apt install -y python-is-python3 && \
    apt-get install -y libgl1 python3-pip

RUN rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip

RUN pip install --upgrade pip

COPY llm-for-geo/requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENV CUDA_HOME=/usr/local/cuda

ENV FORCE_CUDA="1"

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY llm-for-geo /opt/program
# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
WORKDIR /opt/program

