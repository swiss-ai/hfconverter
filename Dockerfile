WORKDIR /workspace

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
ARG TRANSFORMERS_BRANCH=aimv2-fix
ARG MEGATRON_BRANCH=main

FROM ${BASE_IMAGE}

# Install dependencies.
RUN pip install --no-deps --no-build-isolation git+https://github.com/nickjbrowning/XIELU.git@main

RUN git clone https://github.com/swiss-ai/transformers.git && \
    cd transformers && \
    git checkout $TRANSFORMERS_BRANCH && \
    pip install -e . && \
    cd ..

RUN git clone https://github.com/swiss-ai/Megatron-LM.git && \
    cd Megatron-LM && \
    git checkout $MEGATRON_BRANCH && \
    cd ..

RUN pip install nvidia-modelopt==0.27.0