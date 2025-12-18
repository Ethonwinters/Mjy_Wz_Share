(meshgnn) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"                                                                                                                                                                                                                 
2.4.1                                                                                                                                                                                                                                                                                                                                                                                                                            
12.1                                                                                                                                                                                                                                                                                                                                                                                                                             
True    


pip install numpy matplotlib packaging pillow tqdm tensorboard 
pip install opencv-python
pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

pip install torch-geometric



(meshgnn) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
Command 'aria2c' not found, but can be installed with:
sudo snap install aria2c  # version 1.35.0, or
sudo apt  install aria2   # version 1.36.0-1
See 'snap info aria2c' for additional versions.

(meshgnn) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ sudo apt install aria2 -y
E: dpkg was interrupted, you must manually run 'sudo dpkg --configure -a' to correct the problem. 




mkdir -p data
wget https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -P data
wget https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -P data
wget https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord  -P data

conda create -n tfrecord python=3.7 -y
conda activate tfrecord
pip install "tensorflow<1.15"
pip install numpy packaging


meshgnn) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ python train.py
Simulator model initialized
Optimizer initialized
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29950/29950 [2:29:03<00:00,  3.35it/s]
Epoch 1/100 Train Loss: 1.26e-01 Valid Loss: 3.73e-03
-> New best model saved at epoch 1 with valid loss 3.73e-03
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29950/29950 [2:23:40<00:00,  3.47it/s]
Epoch 2/100 Train Loss: 4.56e-02 Valid Loss: 2.87e-03
  -> New best model saved at epoch 2 with valid loss 2.87e-03
  ..........

(base) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ nvidia-smi
Thu Dec 18 17:05:06 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.154.05             Driver Version: 535.154.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX 5880 Ada Gene...    Off | 00000000:52:00.0  On |                    0 |
| 62%   81C    P2             271W / 285W |  23135MiB / 46068MiB |     94%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2831      G   /usr/lib/xorg/Xorg                          698MiB |
|    0   N/A  N/A      2978      G   /usr/bin/gnome-shell                        143MiB |
|    0   N/A  N/A      3170      G   ...in/bin/sunloginclient --cmd=autorun       13MiB |
|    0   N/A  N/A      3367      G   ...) Chrome/58.0.3029.81 Safari/537.36        7MiB |
|    0   N/A  N/A      3470      G   ...en=45586B388FD74A3ADD009CBA1BCBF6D3        6MiB |
|    0   N/A  N/A      4330      G   ...irefox/6700/usr/lib/firefox/firefox      374MiB |
|    0   N/A  N/A     37582      C   python                                    21708MiB |
|    0   N/A  N/A     38064      G   /proc/self/exe                              143MiB |
