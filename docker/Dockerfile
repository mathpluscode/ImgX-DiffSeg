# Adapted from https://github.com/deepmind/alphafold/blob/main/docker/Dockerfile
# If changing CUDA/CUDNN versions, also update the corresponding versions in
# the second last command
ARG CUDA=11.8.0
ARG CUDNN=8.6.0
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04

# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA
ARG CUDNN

ARG HOST_UID
ARG HOST_GID

ENV USER=app

# Ensure ARGs are sets
RUN test -n "$HOST_UID" && test -n "$HOST_GID"

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

# Create group and user, add -f to skip the command without error if it exists already
RUN groupadd --force --gid $HOST_GID $USER && \
    useradd -r -m --uid $HOST_UID --gid $HOST_GID $USER

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        vim \
        unzip \
        build-essential \
        cmake \
        python3-opencv \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        git \
        tzdata \
        wget \
        make \
    && rm -rf /var/lib/apt/lists/*

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
RUN echo $'#!/bin/bash\nldconfig'

RUN mkdir -p /${USER}/tmp
RUN mkdir -p /${USER}/ImgX
RUN mkdir -p /${USER}/tensorflow_datasets
RUN chgrp -R ${USER} /${USER} && \
    chmod -R g+rwx /${USER} && \
    chown -R ${USER} /${USER}

USER ${USER}

# Install Miniconda package manager.
RUN wget -q -P /${USER}/tmp  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash /${USER}/tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /${USER}/conda
RUN rm /${USER}/tmp/Miniconda3-latest-Linux-x86_64.sh

# https://anaconda.org/nvidia/cuda-toolkit
ENV PATH="/${USER}/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -n base conda-libmamba-solver \
    && conda config --set solver libmamba \
    && conda install -y --channel "nvidia/label/cuda-${CUDA}" cuda-toolkit \
    && conda install -y -c conda-forge \
      pip \
      python=3.9

# Install pip packages.
COPY docker/requirements.txt /${USER}/requirements.txt

RUN /${USER}/conda/bin/pip3 install --upgrade pip \
    && /${USER}/conda/bin/pip3 install \
      jax==0.4.20 \
      jaxlib==0.4.20+cuda11.cudnn86 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && /${USER}/conda/bin/pip3 install tensorflow-cpu==2.14.0 \
    && /${USER}/conda/bin/pip3 install -r /${USER}/requirements.txt

RUN git config --global --add safe.directory /${USER}/ImgX

WORKDIR /${USER}/ImgX
