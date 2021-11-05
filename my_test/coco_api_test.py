from pycocotools.coco import COCO

anno_path = '/opt/tiger/minist/datasets/coco/cocosplit/cocosplit/datasplit/5k.json'
anno_path = '/opt/tiger/minist/datasets/coco/cocosplit_self/seed0/full_box_5shot_trainval.json'
# anno_path = '/opt/tiger/minist/datasets/coco/cocosplit/seed0/full_box_1shot_airplane_trainval.json'
# anno_path = '/opt/tiger/minist/datasets/coco/cocosplit/seed0/full_box_1shot_bed_trainval.json'

coco = COCO(anno_path)
print(len(coco.anns))
# count = 0
# for k, v in coco.catToImgs.items():
#     if len(v) == 1 :
#         # print(k, len(v))
#         count= count + 1
# print(count)

img_ids = list(sorted(coco.imgs.keys()))

# print(type(coco_ids), type(coco.imgs)) # list, dict
# for id in coco_ids:
# cat = coco.loadCats(img_ids[:10])
print(len(img_ids))
# img_id = img_ids[0]
# ann_ids = coco.getAnnIds(imgIds=img_id)
# target = coco.loadAnns(ann_ids)
# print(target)

# cat = coco.getCatIds(img_ids = img_id)
# print(cat)
index = 0
img_id = img_ids[index]
ann_ids = coco.getAnnIds(imgIds=img_id)
print(type(ann_ids), len(ann_ids))
target = coco.loadAnns(ann_ids)
print(type(target))
print(len(target))
print(target[0])
# print(target['labels'])


