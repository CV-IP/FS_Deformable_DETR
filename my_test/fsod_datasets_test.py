# from datasets.FSOD_settings.get_fsod_data_matadata import _get_builtin_metadata

from datasets.torchvision_datasets.coco import FsCocoDetection 

if __name__ == "__main__":
    anno_path = '/opt/tiger/minist/datasets/coco/cocosplit/cocosplit/datasplit/5k.json'
    img_root = '/opt/tiger/minist/datasets/coco/val2014'
    coco5k = FsCocoDetection(img_root, anno_path, dataset_name='coco_base')
    print(len(coco5k))
    img, target = coco5k[0]
    # print(img)
    print(len(target))

    # coco_metadata = _get_builtin_metadata('coco_fewshot')
    # print(coco_metadata['all_classes'])
    # print(coco_metadata['all_dataset_id_to_contiguous_id'])
    # print(coco_metadata['novel_classes'])
    # print(coco_metadata['novel_dataset_id_to_contiguous_id'])
    # print(coco_metadata['base_classes'])
    # print(coco_metadata['base_dataset_id_to_contiguous_id'])