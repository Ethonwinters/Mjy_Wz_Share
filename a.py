(base) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~$ aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
Caught Error while parsing environment variable 'all_proxy'
Exception: [AbstractOptionHandler.cc:69] errorCode=28 We encountered a problem while processing the option '--all-proxy'.
  -> [OptionHandlerImpl.cc:520] errorCode=1 unrecognized proxy format


12/19 09:53:46 [NOTICE] Downloading 1 item(s)

12/19 09:53:47 [ERROR] CUID#7 - Download aborted. URI=https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord
Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord
  -> [SocketCore.cc:1018] errorCode=1 SSL/TLS handshake failure: The TLS connection was non-properly terminated.

12/19 09:53:47 [NOTICE] Download GID#bc520c6a1a312334 not complete: 

Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
bc520c|ERR |       0B/s|https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord

Status Legend:
(ERR):error occurred.

aria2 will resume download if the transfer is restarted.
If there are any errors, then see the log file. See '-l' option in help/man page for details.


unset all_proxy
unset ALL_PROXY
unset http_proxy
unset https_proxy


import numpy as np

node_type = data['node_type'][0, :, 0]
mesh_pos = data['mesh_pos'][0]

for t in range(7):
    idx = np.where(node_type == t)[0]
    print(t, len(idx))



import matplotlib.pyplot as plt

for t in range(7):
    idx = np.where(node_type == t)[0]
    if len(idx) > 0:
        plt.scatter(mesh_pos[idx,0], mesh_pos[idx,1], s=2, label=str(t))

plt.legend()
plt.axis("equal")
plt.show()
