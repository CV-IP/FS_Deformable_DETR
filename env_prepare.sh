# pwd: minist
cd ..
mkdir packages
cd packages
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.05-Linux-x86_64.sh

conda create -n deformable_detr python=3.7 pip
conda activate deformable_detr

conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt


git remote rm origin
git remote add origin git@github.com:StephenStorm/FS_Deformable_DETR.git