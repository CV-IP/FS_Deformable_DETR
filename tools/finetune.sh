#! /usr/bin/python3
for seed in 0 1 2 3 4 5 6 7 8 9
do
    # for shot in  2 3 5 10 30
    for shot in 10 30
    do
        GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 configs/r50_deformable_detr_plus_iterative_bbox_refinement.sh \
        --lr_backbone 0 --freeze_transformer --num_queries 100 \
        --num_classes 20    \
        --filter_kind novel \
        --dataset_name coco_all_seed_${seed}_${shot}_shot     \
        --eval_dataset coco_novel   \
        --output_dir exps/coco_novel_resnet50_q100_onestage_bboxRefine/seed${seed}_${shot}shot     \
        --epochs 50 --batch_size 4 --lr 2e-3 --lr_drop 40 \
        --resume surgery_model/base_resnet50_q100_onestage_bboxRefine_20_classes.pth \
        > log/coco_novel_resnet50_q100_onestage_bboxRefine/seed${seed}_${shot}shot.log
    done
done
