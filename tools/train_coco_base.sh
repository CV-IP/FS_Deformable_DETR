GPUS_PER_NODE=8 nohup ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh --num_classes 60 --batch_size 4 --two_stage --with_box_refine --resume exps/r50_deformable_detr/checkpoint0004.pth >> nohup_coco_base_1103.out 2>&1 &


GPUS_PER_NODE=8 nohup ./tools/run_dist_launch.sh 8 configs/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh --num_classes 60 --batch_size 4 --resume exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/eval/latest.pth>> nohup_coco_base_1105_300query.out 2>&1 &

/opt/tiger/minist/datasets/coco/cocosplit_self/seed0/full_box_1shot_trainval.json


GPUS_PER_NODE=8 nohup ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh --num_classes 80 --batch_size 4 --two_stage --with_box_refine --resume exps/r50_deformable_detr/checkpoint0004.pth >> nohup_coco_base_1103.out 2>&1 &

# seed_0_1_shot 16.5% epoch35



GPUS_PER_NODE=8  ./tools/run_dist_launch.sh 8 configs/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh \
    --eval_dataset coco_all \
    --output_dir exps/coco_seed_0_1_shot \
    --lr_backbone 0 \
    --num_queries 100 \
    --num_classes 80 --batch_size 4 \
    --dataset_name coco_novel_seed_0_1_shot \
    --epochs 50 \
    --lr 2e-4 \
    --lr_drop 40 \
    --resume surgery_model/checkpoint0049_base_441_100q.pth
    
    >> nohup_coco_base_1105_seed0_1shot.out 2>&1 &