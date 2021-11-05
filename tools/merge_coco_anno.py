import os
import json


coco_split_dir = '/opt/tiger/minist/datasets/coco/cocosplit'
new_split_dir = coco_split_dir + '_self'
if not os.path.exists(new_split_dir) :
    os.makedirs(new_split_dir)

images = []
annotations = []
categories = None
licenses = None
info = None

for sub_dir in os.listdir(coco_split_dir):
    if 'seed' in sub_dir:
        save_dir = os.path.join(new_split_dir, sub_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        json_files = os.listdir(os.path.join(coco_split_dir, sub_dir))
        for shot in [1, 2, 3, 5, 10, 30] :
            file_name = 'full_box_{}shot_trainval.json'.format(shot)
            images = []
            annotations = []
            categories = None
            licenses = None
            info = None
            wr_file = open(os.path.join(save_dir, file_name), 'w') # need close !!!
            for json_file in json_files:
                if str(shot)+'shot' in json_file :
                    json_file_abs_path = os.path.join(coco_split_dir, sub_dir,json_file)
                    line = json.load(open(json_file_abs_path, 'r'))
                    if licenses is None:
                        licenses = line['licenses']
                    if categories is None:
                        categories = line['categories']
                    if info is None:
                        info = line['info']
                    annotation = line['annotations']
                    image = line['images']
                    annotations.extend(annotation)

                    img_id_with_anno = []
                    for anno in annotation:
                        img_id_with_anno.append(anno['image_id'])
                    
                    filter_imgs = []
                    for img in image:
                        if img['id'] in img_id_with_anno:
                            filter_imgs.append(img)


                    images.extend(filter_imgs)
                    
            save_json = {'info':info, 'licenses' : licenses, 'categories' : categories, 'images':images, 'annotations' : annotations}
            json.dump(save_json, wr_file)
            wr_file.close()
        


