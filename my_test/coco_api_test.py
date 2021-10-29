from pycocotools.coco import COCO

anno_path = '/opt/tiger/minist/datasets/coco/cocosplit/cocosplit/datasplit/5k.json'

coco = COCO(anno_path)



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


