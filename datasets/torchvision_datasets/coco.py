# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
from sys import meta_path, exit
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO


from datasets.FSOD_settings.get_fsod_data_matadata import _get_builtin_metadata


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)



class FsCocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1, dataset_name = 'coco_all'):
        '''
        dataset_name = 'coco_base, coco_all, coco_{novel / all}_seed_{s}_{k}_shot'

        '''
        
        super(FsCocoDetection, self).__init__(root, transforms, transform, target_transform)
        self.dataset_name = dataset_name
        self.metadata = _get_builtin_metadata('coco_fewshot')
        id_map_key = 'all_dataset_id_to_contiguous_id'
        if 'shot' in self.dataset_name :
            if 'novel' in self.dataset_name:
                # coco_novel_seed_{s}_{k}_shot
                id_map_key = 'novel_dataset_id_to_contiguous_id'
        elif self.dataset_name == 'coco_base':
            id_map_key = 'base_dataset_id_to_contiguous_id'
        
        self.id_map = self.metadata[id_map_key]
        print('id map')
        print(self.id_map)

        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        
        
        img_ids = list(sorted(self.coco.imgs.keys()))
        # ann_ids = self.coco.getAnnIds(imgIds=img_ids)

        self.ids = []
        # assert len(img_ids) == len(ann_ids), 'length of img_ids : {},  ann_ids : {},  Not Equal !!!'.format(len(img_ids), len(ann_ids))
        # if self.dataset_name == 'coco_base' :
        if 'shot' in self.dataset_name:
            # 不需要对训练集进行过滤
            self.ids = list(sorted(self.coco.imgs.keys()))
        else :
            # 需要过滤，留下只是base或者是novel的类别。
            for i, img_id in enumerate(img_ids):
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                positive = False # 该图片是否包含对应训练集中的类， all / base / novel
                for ann in anns:
                    if ann['category_id'] in self.id_map:
                        positive = True
                        break
                if positive or len(ann_ids) == 0:
                    self.ids.append(img_ids[i]) 

        print('{} dataset nums : {}'.format(self.dataset_name, len(self.ids)))

        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target_all = coco.loadAnns(ann_ids)
        target = []
        for anno in target_all:
            if anno["category_id"] in self.id_map:
                # ori class_id 2 continious class_id
                anno["category_id"] = self.id_map[anno["category_id"]]
                target.append(anno)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)