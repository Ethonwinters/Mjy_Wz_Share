(meshgnn) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"                                                                                                                                                                                                                 
2.4.1                                                                                                                                                                                                                                                                                                                                                                                                                            
12.1                                                                                                                                                                                                                                                                                                                                                                                                                             
True    

# 1. 创建一个名为 rectified_flow 的全新 conda 环境
# 推荐使用 Python 3.10，它目前在深度学习生态中兼容性最佳
conda create -n rectified_flow python=3.10 -y

# 2. 激活刚刚创建的环境
conda activate rectified_flow

# 3. 安装 PyTorch、TorchVision 及其 CUDA 依赖（强烈建议使用 GPU 运行）
# 注意：这里以目前最常用的 CUDA 12.1 版本为例。
# 如果你的显卡/驱动较老，需要 CUDA 11.8，请将下面的 pytorch-cuda=12.1 改为 pytorch-cuda=11.8
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. 通过 pip 安装核心库以及 Trainer 依赖的 accelerate
# (pip 通常会自动帮你把 einops 等 lucidrains 常用底层依赖一起装好)
pip install rectified-flow-pytorch accelerate

(rectified_flow_3) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/rectified-flow-pytorch$ python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" 
2.11.0+cu130
13.0
/home/adminstrator/miniforge3/envs/rectified_flow_3/lib/python3.10/site-packages/torch/cuda/__init__.py:180: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12020). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
False



(rectified_flow_2) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/rectified-flow-pytorch$ pip install rectified-flow-pytorch
找不到命令 “pip”，但可以通过以下软件包安装它：
sudo apt install python3-pip

