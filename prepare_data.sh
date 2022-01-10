# mkdir datasets
# cd datasets
# mkdir coco
# cd coco

# hdfs dfs -get  hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/exps/surgery_model
mkdir datasets
cd datasets
mkdir coco
cd coco
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/train2014.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/val2014.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/train2017.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/val2017.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/annotations_trainval2017.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/exps/cocosplit_self

unzip train2014
unzip val2014
unzip train2017
unzip val2017
unzip annotations_trainval2017.zip
mv val2014/* train2014
mv train2014 JPEG


cd /opt/tiger/minist/FS_Deformable_DETR
mkdir data
cd data
ln -s /opt/tiger/minist/datasets/coco coco