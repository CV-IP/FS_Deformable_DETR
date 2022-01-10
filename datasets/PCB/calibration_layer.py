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
from torch.utils.data import DataLoader
import util.misc as utils
# from defrcn.dataloader import build_detection_test_loader



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
        if args.distributed:
            if args.cache_mode:
                sampler_train = samplers.NodeDistributedSampler(self.dataset_train, shuffle=False)

            else:
                sampler_train = samplers.DistributedSampler(self.dataset_train, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(self.dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        self.dataloader = DataLoader(self.dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)


        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
        self.prototypes = self.build_prototypes()

        self.exclude_cls = self.clsid_filter()

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

        all_features, all_labels = [], []
        for index in range(len(self.dataset_train)):
            img, target = self.dataset_train[index]
            boxes = target['boxes']
            all_labels.append(target['labels'].cpu().data)

            # extract roi features
            features = self.extract_roi_features(img, boxes)
            all_features.append(features.cpu().data)

        # concat
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
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

        return prototypes_dict

    def extract_roi_features(self, imgs, boxes):
        """
        
            
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

    def execute_calibration(self, inputs, dts):
        '''
        inputs: imgs, NCHW, input for model. model(inputs)
        dts: res after postprocess in deformable_detr.py
        dts：[scores, labels, bboxes]
        '''

        # img = cv2.imread(inputs[0]['file_name'])

        ileft = (dts[0]['instances'].scores > self.args.pcb_upper).sum()
        iright = (dts[0]['instances'].scores > self.args.pcb_lower).sum()
        assert ileft <= iright


        # boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]
        boxes = [box for score, label, box in dts]

        # features = self.extract_roi_features(img, boxes)
        features = self.extract_roi_features(inputs, boxes)

        for i in range(ileft, iright):
            tmp_class = int(dts[0]['instances'].pred_classes[i])
            if tmp_class in self.exclude_cls:
                continue
            tmp_cos = cosine_similarity(features[i - ileft].cpu().data.numpy().reshape((1, -1)),
                                        self.prototypes[tmp_class].cpu().data.numpy())[0][0]
            dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
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
