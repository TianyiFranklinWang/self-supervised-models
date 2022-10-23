#!/bin/bash
OMP_NUM_THREADS='6' torchrun \
                      --nnodes=1 \
                      --nproc_per_node="$1" \
                      main_pretrain.py
