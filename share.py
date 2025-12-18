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
Traceback (most recent call last):
  File "parse_tfrecord.py", line 8, in <module>
    import tensorflow as tf
  File "/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/__init__.py", line 28, in <module>
    from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
  File "/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/python/__init__.py", line 52, in <module>
    from tensorflow.core.framework.graph_pb2 import *
  File "/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/core/framework/graph_pb2.py", line 16, in <module>
    from tensorflow.core.framework import node_def_pb2 as tensorflow_dot_core_dot_framework_dot_node__def__pb2
  File "/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/core/framework/node_def_pb2.py", line 16, in <module>
    from tensorflow.core.framework import attr_value_pb2 as tensorflow_dot_core_dot_framework_dot_attr__value__pb2
  File "/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/core/framework/attr_value_pb2.py", line 16, in <module>
    from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
  File "/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/core/framework/tensor_pb2.py", line 16, in <module>
    from tensorflow.core.framework import resource_handle_pb2 as tensorflow_dot_core_dot_framework_dot_resource__handle__pb2
  File "/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/tensorflow/core/framework/resource_handle_pb2.py", line 42, in <module>
    serialized_options=None, file=DESCRIPTOR),
  File "/home/adminstrator/miniforge3/envs/tfrecord/lib/python3.7/site-packages/google/protobuf/descriptor.py", line 561, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
