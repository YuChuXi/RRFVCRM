python --version
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
