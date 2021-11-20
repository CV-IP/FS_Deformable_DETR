import os
import torch
import argparse
import math

'''
deformable-detr:
    utils.save_on_master({
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }, checkpoint_path)
'''

def deformable_detr_model_surgery(ckpt_path, save_dir = '/opt/tiger/minist/FS_Deformable_DETR/surgery_model', all_class_nums = 80):
    ckpt = torch.load(ckpt_path)
    ckpt_name = ckpt_path.split('/')[-1].split('.')[0] + '_' + str(all_class_nums) + '_classes.pth'
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    class_embed_bias = torch.ones(all_class_nums) * bias_value
    if 'lr_scheduler' in ckpt:
        del ckpt['lr_scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'epoch' in ckpt:
        ckpt['epoch'] = 0

    # model = ckpt['model']
    for param, weight in ckpt['model'].items():
        if 'class_embed' in param:
            # is_bias =
            print(param, weight.shape) 
            if 'bias' in param :
                new_weight = class_embed_bias
            else :
                new_weight = torch.rand((all_class_nums, weight.shape[1]))
                torch.nn.init.normal_(new_weight, 0, 0.01)
            ckpt['model'][param] = new_weight
    
    save_path = os.path.join(save_dir, ckpt_name)
    torch.save(ckpt, save_path)
    print('save new weight to {}'.format(save_path))
    return save_path


def show_weight_param(ckpt_path):
    ckpt = torch.load(ckpt_path)
    for param, weight in ckpt['model'].items():
        print(param, weight.shape)



if __name__ == '__main__':
    ckpt_path = '/opt/tiger/minist/FS_Deformable_DETR/exps/coco_base_100_q_resnet50_1117_one_stage/best_checkpoint.pth'
    save_dir = '/opt/tiger/minist/FS_Deformable_DETR/surgery_model'
    save_path = deformable_detr_model_surgery(ckpt_path, save_dir, all_class_nums = 80)
    # ckpt_name = ckpt_path.split('/')[-1].split('.')[0] + '_' + str(20) + 'classes.pth'
    show_weight_param(save_path)
    

