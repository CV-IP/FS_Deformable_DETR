import os
import json
from glob import glob



log_dir = '/opt/tiger/bytedetection/tools/FS_Deformable_DETR/log/coco_train_all_eval_novel_resnet50_q100_twostage_bboxRefine_no_freeze'
res_file = os.path.join(log_dir, 'res.json')



seeds = list(range(10))
# shots = [2, 3, 5, 10, 30]
shots = [10, 30]
res_f = open(res_file, 'w') # have to close this file
for shot in shots :
    total_mAP = 0.0
    mAPs = []
    for seed in seeds:
        log_file = 'seed{}_{}shot.log'.format(seed, shot)
        f = open(os.path.join(log_dir, log_file), 'r')
        line = f.readlines()[-1].strip()
        f.close()
        mAP = float(line.split()[-1])
        mAPs.append(mAP)
        total_mAP = total_mAP + mAP

    average_mAP = total_mAP / len(seeds)
    res_dict = {'shot' : shot, 'average mAP' : average_mAP, 'all seeds mAP':mAPs}
    res_f.write(json.dumps(res_dict) + '\n')

res_f.close()



    
    


