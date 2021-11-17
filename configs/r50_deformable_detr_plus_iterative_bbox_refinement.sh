#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement
PY_ARGS=${@:1}

python -u main.py \
    --with_box_refine \
    ${PY_ARGS}
