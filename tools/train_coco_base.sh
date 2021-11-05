GPUS_PER_NODE=8 nohup ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh --num_classes 60 --batch_size 4 --two_stage --with_box_refine --resume exps/r50_deformable_detr/checkpoint0004.pth >> nohup_coco_base_1103.out 2>&1 &


GPUS_PER_NODE=8 nohup ./tools/run_dist_launch.sh 8 configs/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh --num_classes 60 --batch_size 4 >> nohup_coco_base_1103.out 2>&1 &

/opt/tiger/minist/datasets/coco/cocosplit_self/seed0/full_box_1shot_trainval.json


GPUS_PER_NODE=8 nohup ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh --num_classes 80 --batch_size 4 --two_stage --with_box_refine --resume exps/r50_deformable_detr/checkpoint0004.pth >> nohup_coco_base_1103.out 2>&1 &


GPUS_PER_NODE=8 nohup ./tools/run_dist_launch.sh 8 configs/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh --num_classes 80 --batch_size 4 >> nohup_coco_base_1105_seed0_1shot.out 2>&1 &