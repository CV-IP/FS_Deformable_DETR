import sys
import os
import json
from itertools import chain
 
 
START_BOUNDING_BOX_ID = 1           # testing need to change 701
# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {"a": 1, "d": 2}
 
 
def convert(jsonsFile, json_file, imgPath):
    
 
	# ########################################### define the head #################################################
    imgs = os.listdir(imgPath)
    json_dict = {"info":{}, "licenses":[], "images":[], "annotations": [], "categories": []}
	
 
	# ######################################### info, type is dict ################################################
    info = {'description': 'SteelCoils Dataset', 'url': 'None', 'version': '1.0', 'year': 2021,	'contributor': 'Donghao', 'note': 'category d(2) is SteelCoils(钢卷), category a(1) is Scooter(板车)', 'date_created': '2021/03/05'}
    json_dict['info'] = info
	
 
	# ####################################### licenses, type is list ##############################################
    license = {'url': 'None', 'id': 1, 'name': 'None'}
    json_dict['licenses'].append(license)
	
 
	# ###################################### categories, type is list #############################################
    categories = PRE_DEFINE_CATEGORIES
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid , 'name': cate} # no + 1
        json_dict['categories'].append(cat)
 
 
    bnd_id = START_BOUNDING_BOX_ID
    list_fp = os.listdir(jsonsFile)
    for i, line in enumerate(list_fp):
        
        line = line.strip()
        print(f"Processing {line}, {i}/{len(list_fp)}")
 
        json_path = os.path.join(jsonsFile, line)
        with open(json_path, 'r') as load_f:
            load_dict = json.load(load_f)
 
        
		# ###################################### images, type is list #############################################
        name = line.replace(".json","")
        image_id = int(name)
        image_name = name + ".jpg"        
        if image_name not in imgs:
            print (line)
            print(image_name)		
        width, height = load_dict["imageWidth"], load_dict["imageHeight"]
       
        image = {'license': 0, 'file_name': image_name, 'steelcoils_url': 'None', 'height': height, 'width': width, 'date_captured': '1998-02-05 05:02:01', 'flickr_url': 'None', 'id': image_id}
        json_dict['images'].append(image)
        
 
        # ###################################### annotations, type is list #############################################
        
        shapes =  load_dict["shapes"]
        for obj in shapes: 
            
            label = obj['label']
            if label not in categories:
                new_id = len(categories)
                categories[label] = new_id
            category_id = categories[label]
 
            points = obj['points']
            pointsList = list(chain.from_iterable(points))
            seg = [pointsList]
 
            row = pointsList[0::2]
            clu = pointsList[1::2]
            left_top_x = min(row)
            left_top_y = min(clu)
            right_bottom_x = max(row)
            right_bottom_y = max(clu)
            wd = right_bottom_x - left_top_x
            hg = right_bottom_y - left_top_y
		
            ann = {'segmentation': seg, 'area': wd*hg, 'iscrowd': 0, 'image_id': image_id, 'bbox': [left_top_x, left_top_y, wd, hg], 'category_id': category_id, 'id': bnd_id}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
		
		   
	# ######################################### write into local ################################################
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
 
 
if __name__ == '__main__':
    jsonsFile = "E:\\数据集\\SteelCoils\\training\\totaljson"
    imgPath = "E:\\数据集\\SteelCoils\\training\\total"    
    destJson = "E:\\数据集\\SteelCoils\\training\\training_toCOCO_augment.json"
 
    convert(jsonsFile, destJson, imgPath)