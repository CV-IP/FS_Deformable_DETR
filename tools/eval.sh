#! /usr/bin/python3
EXP_DIR=coco_train_all_eval_novel_resnet50_q100_onestage_NobboxRefine_freeze_head_20211229

for seed in 0 1 2 3 4 5 6 7 8 9
do
    # for shot in  2 3 5 10 30
    for shot in 10 30
    do
        GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 configs/r50_deformable_detr.sh \
        --num_queries 100 --num_classes 80    \
        --dataset_name coco_all_seed_${seed}_${shot}_shot     \
        --eval_dataset coco_novel   \
        --output_dir exps/test   \
        --batch_size 4  \
        --resume exps/${EXP_DIR}/seed${seed}_${shot}shot/best_checkpoint.pth \
        --eval --pcb_enable \
        > log/${EXP_DIR}/seed${seed}_${shot}shot_eval_novel_pcb.log
    done
done
