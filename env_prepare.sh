# pwd: minist
export http_proxy=10.20.47.147:3128  https_proxy=10.20.47.147:3128 no_proxy=code.byted.org

git clone -b fs_deformable_detr https://github.com/StephenStorm/FS_Deformable_DETR.git

cd ..
mkdir packages
cd packages
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.05-Linux-x86_64.sh

conda create -n deformable_detr python=3.7 pip
conda activate deformable_detr

conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
pip install opencv-python sklearn


git remote rm origin
git remote add origin git@github.com:StephenStorm/FS_Deformable_DETR.git

hdfs dfs -get  hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/exps/surgery_model
cd surgery_model
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
mkdir data
cd data
ls -s 

mkdir datasets
cd datasets
mkdir coco
cd coco
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/train2014.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/zhanglibin.buaa/datasets/val2014.zip
unzip train2014
unzip val2014
mv val2014/* train2014
mv train2014 JPEG
