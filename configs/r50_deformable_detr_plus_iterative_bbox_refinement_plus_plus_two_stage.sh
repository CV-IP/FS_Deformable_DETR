#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_seed0_1shot
PY_ARGS=${@:1}

python -u main.py \
    --with_box_refine \
    --two_stage \
    ${PY_ARGS}
