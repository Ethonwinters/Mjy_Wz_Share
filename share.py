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

(tfrecord) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ python parse_tfrecord.py
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
WARNING:tensorflow:From parse_tfrecord.py:57: The name tf.enable_resource_variables is deprecated. Please use tf.compat.v1.enable_resource_variables instead.

WARNING:tensorflow:From parse_tfrecord.py:58: The name tf.enable_eager_execution is deprecated. Please use tf.compat.v1.enable_eager_execution instead.

Traceback (most recent call last):
  File "parse_tfrecord.py", line 61, in <module>
    ds = load_dataset(tf_datasetPath, split)
  File "parse_tfrecord.py", line 45, in load_dataset
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
FileNotFoundError: [Errno 2] No such file or directory: 'data/meta.json'
