#! /usr/bin/python3
for seed in 0 1 2 3 4 5 6 7 8 9
do
    # for shot in  2 3 5 10 30
    for shot in 10 30
    do
        GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 configs/r50_deformable_detr.sh \
        --gdl_enable \
        --num_queries 100 \
        --num_classes 80    \
        --dataset_name coco_all_seed_${seed}_${shot}_shot     \
        --eval_dataset coco_novel   \
        --output_dir exps/coco_train_all_eval_novel_resnet50_q100_onestage_NobboxRefine_noFreeze_gdl_20220408/seed${seed}_${shot}shot     \
        --epochs 150 --batch_size 4 --lr 4e-4 --lr_drop 135 \
        --resume surgery_model/base_resnet50_q100_onestage_NobboxRefine_gdl_80_classes.pth \
        > log/coco_train_all_eval_novel_resnet50_q100_onestage_NobboxRefine_noFreeze_gdl_20220408/seed${seed}_${shot}shot.log

    done
done
