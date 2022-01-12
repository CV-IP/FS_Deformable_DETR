import os
from random import shuffle
import cv2
import json
import torch

import numpy as np
from datasets.PCB.poolers import ROIPooler
from datasets.PCB.layers import resnet101, ImageList
from sklearn.metrics.pairwise import cosine_similarity

from datasets import build_dataset
from pathlib import Path
import datasets.samplers as samplers
from torch.utils.data import DataLoader, dataset
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy
# from defrcn.dataloader import build_detection_test_loader

import sys

'''
settings in DeFRCN
_CC.TEST.PCB_ENABLE = False
_CC.TEST.PCB_MODELTYPE = 'resnet'             # res-like
_CC.TEST.PCB_MODELPATH = ""
_CC.TEST.PCB_ALPHA = 0.50
_CC.TEST.PCB_UPPER = 1.0
_CC.TEST.PCB_LOWER = 0.05
'''

class PrototypicalCalibrationBlock:
    '''
    base train do not need PCB

    '''

    def __init__(self, args):
        super().__init__()
        # for fast adaption, not recommendation
        self.args = args
        self.device = torch.device(args.device)
        self.alpha = self.args.pcb_alpha

        self.imagenet_model = self.build_model()
        
        '''
        ## TODO
        dataset_name = args.dataset_name
        seed = dataset_name.split('_')[3]
        shot = dataset_name.split('_')[4]
        assert int(seed) in range(10) and int(shot) in [1,2,3,5,10,30]
        root = Path(args.coco_path)
        img_folder = os.path.join(root, "JPEG")
        ann_file = os.path.join(root, "cocosplit_self","seed"+ seed, "full_box_{}shot_trainval.json".format(shot))
        '''
        
        self.dataset_train = build_dataset(image_set=args.dataset_name + '_val', args=args)
        # if args.distributed:
        #     if args.cache_mode:
        #         sampler_train = samplers.NodeDistributedSampler(self.dataset_train, shuffle=False)

        #     else:
        #         sampler_train = samplers.DistributedSampler(self.dataset_train, shuffle=False)
        # else:
        #     sampler_train = torch.utils.data.RandomSampler(self.dataset_train)
        sampler_train = torch.utils.data.RandomSampler(self.dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.pcb_batch_size, drop_last=True)

        self.dataloader = DataLoader(self.dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)


        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
        self.prototypes = self.build_prototypes()

        self.exclude_cls = self.clsid_filter()
        # self.exclude_cls = []

    def build_model(self):
        
        if self.args.pcb_model_type == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.args.pcb_model_path)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def build_prototypes(self):
        print('start build prototypes ...')

        all_features, all_labels = [], []
        # print(len(self.dataloader))
        for samples, targets in self.dataloader:
            samples = samples.tensors
            '''
            ###！！！ 需要注意的是，此处的box是normalization之后的坐标，[0, 1]， （cx, cy, w, h)
            when batch_size = 2
            [
                {
                    'boxes': tensor([[0.3049, 0.5058, 0.1351, 0.9103]]), 'labels': tensor([20]), 'image_id': tensor([41288]), 
                    'area': tensor([87186.5000]), 'iscrowd': tensor([0]), 'orig_size': tensor([640, 393]), 'size': tensor([1302,  800])
                }, 
                {
                    'boxes': tensor([[0.2349, 0.5081, 0.4007, 0.9482],[0.6682, 0.6131, 0.6604, 0.7394],[0.9473, 0.8674, 0.1054, 0.2653]]), 
                    'labels': tensor([16, 17, 17]), 'image_id': tensor([193781]), 'area': tensor([109713.2656, 199557.7812,  14150.9287]), 
                    'iscrowd': tensor([0, 0, 0]), 'orig_size': tensor([640, 480]), 'size': tensor([1066,  800])
                }
            ]
            '''
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            # boxes = [box_cxcywh_to_xyxy(target['boxes']) for target in targets]
            
            boxes = []
            vis = False # visualization for check
                
            for idx, target in enumerate(targets):

                all_labels.append(target['labels'].cpu().data)

                box = box_cxcywh_to_xyxy(target['boxes'])
                aug_h, aug_w = target['size'][0], target['size'][1]
                scale_t = torch.tensor([aug_w, aug_h, aug_w, aug_h,]).to(self.device)
                box = (box * scale_t)
                boxes.append(box)

                if vis:
                    print(box)
                    img = samples[idx]
                    img = img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).cpu().type(torch.uint8).numpy()
                    print(img.shape, type(img)) # (800, 1199, 3) <class 'numpy.ndarray'>
                    print(aug_h, aug_w)
                    img2 = img.copy()
                    for j in range(box.shape[0]):
                        cv2.rectangle(img2, (int(box[j][0] + 0.5), int(box[j][1] + 0.5)), (int(box[j][2] + 0.5), int(box[j][3] + 0.5)), (0, 0, 255))
                    
                    cv2.imwrite(str(idx)+'.jpg', img2)
            if vis:
                sys.exit()

            # extract roi features
            features = self.extract_roi_features(samples, boxes)
            all_features.append(features.cpu().data)

        
        # print(len(all_features), len(all_labels))
        # concat
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # print(all_features.shape, all_labels.shape)
        assert all_features.shape[0] == all_labels.shape[0]

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))

        prototypes_dict = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)

        print('prototypes length: {} '.format(len(prototypes_dict)))
        # print(prototypes_dict[0].shape) # torch.Size([1, 1000])
        return prototypes_dict

    def extract_roi_features(self, imgs, boxes):
        """
        ###!!! boxes 必须不是归一化后的坐标，不是[0, 1]，而是相对于原图的坐标。！！！ (x1, y1, x2, y2) format
        :param imgs: shape：BCHW
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
            used to construct this module.
        :param boxes: 是否需要归一化？
            box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
            The box coordinates are defined on the original image and
            will be scaled by the `scales` argument of :class:`ROIPooler`.
        :return:
        """
        # conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW
        conv_feature = self.imagenet_model(imgs)[1]  # size: BxCxHxW

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)

        activation_vectors = self.imagenet_model.fc(box_features)

        return activation_vectors

    def execute_calibration(self, inputs, dts, scale):
        '''
        inputs: tensor, imgs, NCHW, input for model. model(inputs)
        dts: res after postprocess in deformable_detr.py
        dts ：[{scores: ,  labels:, boxes:}] list of dict
        original_size: input for postprocess in FS_Deformable_DETR/models/deformable_detr.py 
            用来消除postprocess 中对预测box的scale操作，将相对于origin  尺寸的box变换为相对于aug后，也就是batch_size的坐标。
        '''

        # boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]
        # boxes = [s_l_b['boxes'] for s_l_b in dts]
        assert len(dts) == inputs.shape[0], 'len(dts) != inputs.shape, len(dts): {}, inputs.shape: {}'.format(len(dts), inputs.shape)
        assert scale.shape[0] == inputs.shape[0]
        boxes = []
        # batch_h, batch_w = inputs.shape[2:]
        # print(batch_h, batch_w)
        scale = scale.flip(1).repeat(1, 2)
        
        vis = False # for debug
        for i in range(len(dts)):
            # scale = (torch.tensor([batch_h, batch_w]).to(self.device) / original_size[i]).flip(0) # [h, w] -> [w, h]
            # scale = scale.repeat(1, 2) # (h, w) ->(h, w, h, w)
            # print(scale)
            box = torch.clamp(dts[i]['boxes'], min=0) * scale[i]
            boxes.append(box)
            if vis:
                print(box[:5])
                img = inputs[i].clone()
                img = img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).cpu().type(torch.uint8).numpy()
                print(img.shape, type(img)) # (800, 1199, 3) <class 'numpy.ndarray'>
                img2 = img.copy()
                # for j in range(box.shape[0]):
                for j in range(30):
                    cv2.rectangle(img2, (int(box[j][0] + 0.5), int(box[j][1] + 0.5)), (int(box[j][2] + 0.5), int(box[j][3] + 0.5)), (0, 0, 255))
                
                cv2.imwrite(str(i)+'_calibrate.jpg', img2)
        if vis:
            sys.exit()

        # print(inputs.tensors.shape, len(boxes))
        features = self.extract_roi_features(inputs, boxes)
        # print(boxes[0].shape[0] + boxes[1].shape[0]) 
        # print('inputs shape: {}, dts length:{}, features shape : {}'.format(inputs.tensors.shape, len(dts), features.shape))
        # return dts
        # print(self.prototypes.shape)
        for j in range(len(dts)): # batch-size
            # print(dts[j]['scores']) # 从大到小的顺序
            
            ileft = (dts[j]['scores'] > self.args.pcb_upper).sum()
            iright = (dts[j]['scores'] > self.args.pcb_lower).sum()
            # print(ileft, iright)
            assert ileft <= iright, 'ileft({}) > iright({})'.format(ileft, iright)
            for i in range(ileft, iright):
                tmp_class = int(dts[j]['labels'][i])
                if tmp_class in self.exclude_cls:
                    continue
                tmp_cos = cosine_similarity(features[j * self.args.num_queries + i - ileft].cpu().data.numpy().reshape((1, -1)),
                                            self.prototypes[tmp_class].cpu().data.numpy())[0][0]
                dts[j]['scores'][i] = dts[j]['scores'][i] * self.alpha + tmp_cos * (1 - self.alpha)
        
        return dts

    def clsid_filter(self):
        eval_dataset = self.args.eval_dataset
        exclude_ids = []
        # 只测试base的性能。不需要顾虑，所有都类别都进行评估
        # 只有shot数据集才会使用PCB，因此base这种情况是不可能出现的
        # if 'base' in eval_dataset :
        #     pass

        # if 'novel' in eval_dataset:
            # 两种情况，一种是只有novel类别参与训练。0-19，这个时候不需要过滤，所有都需要使用PCB
            # 还有一种情况是使用all class进行训练，only eval on novel classes ,这个时候只需要矫正novel class
        # if 'all' in eval_dataset:
            # only calibrate novel classes
        ### 综上分析：
        if 'coco' in self.args.dataset_name and self.args.num_classes == 80:
            exclude_ids = list(range(0, 60))
            # exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            #                 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
            #                 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
            #                 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
        return exclude_ids

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
