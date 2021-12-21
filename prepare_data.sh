# mkdir datasets
# cd datasets
# mkdir coco
# cd coco
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/train2014.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/val2014.zip
unzip train2014
unzip val2014
mv val2014/* train2014
mv train2014 JPEG