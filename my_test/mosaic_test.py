# from datasets.FSOD_settings.get_fsod_data_matadata import _get_builtin_metadata

from datasets.coco import CocoDetection, make_coco_transforms
from datasets.mosaic import MosaicDetection
from datasets.mosaic_utils import TrainTransform
import torch 
import torchvision.transforms.functional as F


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target

if __name__ == "__main__":
    anno_path = 'data/coco/cocosplit_self/datasplit/5k.json'
    img_root = 'data/coco/JPEG'
    transforms = make_coco_transforms('train')
    coco5k = CocoDetection(img_root, anno_path, transforms = None, return_masks = False, dataset_name='coco_base')
    image, target = coco5k[0]
    # print(image.shape)
    # print(target)
    mosaic_coco5k = MosaicDetection(coco5k, preproc=TrainTransform(max_labels=200), mixup_prob=0 )
    # print(len(coco5k))
    img, target = mosaic_coco5k[0]
    # print(type(img), type(target))
    # print(img.shape, target.shape)
    
    print(img.shape)
    for key, value in target.items():
        print(key, type(value))

    # bboxes = target['boxes']
    # classes = target['labels']
    # _label = torch.cat([bboxes, classes.view(-1, 1).float()], dim = 1)
    # print(_label)
    # print(img)
    # print(len(target))

    # coco_metadata = _get_builtin_metadata('coco_fewshot')
    # print(coco_metadata['all_classes'])
    # print(coco_metadata['all_dataset_id_to_contiguous_id'])
    # print(coco_metadata['novel_classes'])
    # print(coco_metadata['novel_dataset_id_to_contiguous_id'])
    # print(coco_metadata['base_classes'])
    # print(coco_metadata['base_dataset_id_to_contiguous_id'])