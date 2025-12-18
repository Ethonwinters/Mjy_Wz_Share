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

