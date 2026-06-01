(base) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:16:58_PDT_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
(base) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~$ nvidia-smi
Mon Jun  1 15:45:37 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.154.05             Driver Version: 535.154.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX 5880 Ada Gene...    Off | 00000000:52:00.0  On |                    0 |
| 30%   35C    P8              12W / 285W |    337MiB / 46068MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                          148MiB |
|    0   N/A  N/A      3021      G   /usr/bin/gnome-shell                        142MiB |
|    0   N/A  N/A      3210      G   ...in/bin/sunloginclient --cmd=autorun       13MiB |
|    0   N/A  N/A      3424      G   ...) Chrome/58.0.3029.81 Safari/537.36        7MiB |
|    0   N/A  N/A      3516      G   ...en=9AB17541D0E4E22C6E1E17B801AE2547        6MiB |
+---------------------------------------------------------------------------------------+

  https://sglox4pz.doggygosubs.com:8443/api/v1/client/3d8456099ded6d52391c10abf515ca1f

下面这份流程按你的机器来写：Ubuntu 22.04、RTX 5880 Ada、Driver 535.154.05、系统 CUDA 12.2。关键点先说清楚：PhysicsNeMo 当前 CUDA 12 后端对应的是 PyTorch CUDA 12.8，不是 cu122；你的 R535 驱动满足 CUDA 12.x minor compatibility 的最低要求，但如果后面遇到 cudaErrorCallRequiresNewerDriver，就升级驱动到 R570+。NVIDIA 文档当前明确列的系统是 Ubuntu 24.04，22.04 通常可跑，但不算最新文档的明示目标。来源我核对了 PhysicsNeMo 安装文档、README、PyPI 和 CUDA compatibility 文档。(docs.nvidia.com) (docs.nvidia.com) (pypi.org) (docs.nvidia.com)

1. 建 conda 环境

conda create -n geotransolver python=3.12 -y
conda activate geotransolver
python -m pip install --upgrade pip setuptools wheel
2. 安装系统依赖

sudo apt update
sudo apt install -y git build-essential cmake ninja-build libgl1 libglib2.0-0
3. 安装 PhysicsNeMo CUDA 12 后端

pip install --extra-index-url https://pypi.nvidia.com "nvidia-physicsnemo[cu12,nn-extras]"
PhysicsNeMo PyPI 最新包要求 Python >=3.11,<3.14，所以这里用 Python 3.12。(pypi.org)

4. 克隆示例并安装 GeoTransolver 额外依赖

cd ~
git clone https://github.com/NVIDIA/physicsnemo.git
cd ~/physicsnemo/examples/cfd/external_aerodynamics/transformer_models

pip install --extra-index-url https://pypi.nvidia.com -r requirements.txt
该 README 说明这个例子需要 requirements.txt，并要求 PhysicsNeMo 25.11 或更高；当前 PyPI 2.1.0 满足“更高”的要求。README 里的配置名包括 geotransolver_surface、geotransolver_volume。(raw.githubusercontent.com) (github.com) (github.com)

5. 验证 GPU / PyTorch / PhysicsNeMo

python - <<'PY'
import torch, physicsnemo
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("physicsnemo:", physicsnemo.__version__)
x = torch.randn(1024, 1024, device="cuda")
print((x @ x).mean())
PY
如果这里 cuda available: True，环境主体就对了。

6. 准备 DrivAerML 数据

cd ~
git clone https://github.com/NVIDIA/physicsnemo-curator.git
cd ~/physicsnemo-curator
pip install "physicsnemo-curator[mesh,loky]"
下载小样本先测试：

cd ~/physicsnemo-curator/examples/external_aerodynamics
chmod +x download_hugging_face_dataset.sh
./download_hugging_face_dataset.sh -d ~/data/drivaerml_raw -s 1 -e 5
跑 Curator，生成 Zarr：

cd ~/physicsnemo-curator
python run_etl.py \
  --config-dir=examples/external_aerodynamics/config \
  --config-name=external_aero_etl_drivaerml \
  etl.source.input_dir=~/data/drivaerml_raw \
  etl.sink.output_dir=~/data/drivaerml_processed_surface \
  etl.common.model_type=surface
Curator README 明确说明该流程把 DrivAerML 的 STL/VTP/VTU 等 CFD 数据转成训练用 Zarr/NumPy，并且适用于 DoMINO 和 Transolver/GeoTransolver 这类模型。(github.com)

7. 计算归一化文件

cd ~/physicsnemo/examples/cfd/external_aerodynamics/transformer_models
mkdir -p normalizations runs

python src/compute_normalizations.py \
  --config-name geotransolver_surface \
  data.train.data_path=~/data/drivaerml_processed_surface/train \
  data.normalization_dir=$PWD/normalizations
如果你的 Curator 输出目录没有自动分 train/val，就先看一下：

find ~/data/drivaerml_processed_surface -maxdepth 2 -type d
然后把 data.train.data_path、data.val.data_path 改成实际 Zarr 目录。

8. 先做 1 epoch 冒烟测试

python train.py \
  --config-name geotransolver_surface \
  data.train.data_path=~/data/drivaerml_processed_surface/train \
  data.val.data_path=~/data/drivaerml_processed_surface/val \
  data.normalization_dir=$PWD/normalizations \
  data.resolution=20000 \
  training.num_epochs=1 \
  output_dir=$PWD/runs \
  run_id=geotransolver/surface/smoke
9. 正式训练 GeoTransolver surface

python train.py \
  --config-name geotransolver_surface \
  data.train.data_path=~/data/drivaerml_processed_surface/train \
  data.val.data_path=~/data/drivaerml_processed_surface/val \
  data.normalization_dir=$PWD/normalizations \
  data.resolution=200000 \
  output_dir=$PWD/runs \
  run_id=geotransolver/surface/bq
你的 46GB 显存可以先试默认 200000；如果 OOM，降到 100000 或 50000。README 也说明可用 data.resolution=N 控制每张 GPU 使用的点数。(raw.githubusercontent.com)

10. 推理

python src/inference_on_zarr.py \
  --config-name geotransolver_surface \
  run_id=$PWD/runs/geotransolver/surface/bq \
  data.val.data_path=~/data/drivaerml_processed_surface/val \
  data.normalization_dir=$PWD/normalizations \
  data.return_mesh_features=true
常见坑：不要装 pytorch-cuda=12.2，这里走 pip 轮子；不要把 nvcc -V 的 12.2 当成 PyTorch 必须匹配的版本；README 里模型名有时写 GeoTranSolver，但 Hydra 配置名是小写 geotransolver_surface / geotransolver_volume。





4:00 PM
