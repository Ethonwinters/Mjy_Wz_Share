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

(meshgnn) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
Caught Error while parsing environment variable 'all_proxy'
Exception: [AbstractOptionHandler.cc:69] errorCode=28 We encountered a problem while processing the option '--all-proxy'.
  -> [OptionHandlerImpl.cc:520] errorCode=1 unrecognized proxy format


12/18 09:55:14 [NOTICE] Downloading 1 item(s)

12/18 09:55:14 [ERROR] CUID#7 - Download aborted. URI=https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord
Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord
  -> [SocketCore.cc:1018] errorCode=1 SSL/TLS handshake failure: The TLS connection was non-properly terminated.

12/18 09:55:14 [NOTICE] Download GID#399894f2041ffe74 not complete: 

Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
399894|ERR |       0B/s|https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord

Status Legend:
(ERR):error occurred.

aria2 will resume download if the transfer is restarted.
If there are any errors, then see the log file. See '-l' option in help/man page for details.



mkdir -p data
wget https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -P data
wget https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -P data
wget https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord  -P data
