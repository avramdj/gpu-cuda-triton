#!/bin/bash

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

torchrun --nproc-per-node 2 "$@"
