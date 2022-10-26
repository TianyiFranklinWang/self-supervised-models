#!/bin/bash
OMP_NUM_THREADS="$1" torchrun \
                      --nnodes=1 \
                      --nproc_per_node="$2" \
                      main_pretrain.py
