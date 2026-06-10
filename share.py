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



# 创建数据文件夹
mkdir -p data/flowers

# 下载牛津花卉数据集的图片压缩包
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

# 解压到指定文件夹
tar -xzf 102flowers.tgz -C data/flowers

# 此时，你的图片会存放在 ./data/flowers/jpg 路径下

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"



conda create -n physicsnemo python=3.11 -y
conda activate physicsnemo

python -m pip install --upgrade pip setuptools wheel
pip install "nvidia-physicsnemo[cu12,nn-extras]"

1. 进入你的 PhysicsNeMo 环境

假设你之前创建的环境叫 physicsnemo：

conda activate physicsnemo

检查版本：

python -c "import physicsnemo; print(physicsnemo.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

README 要求这个 example 使用 PhysicsNeMo 25.11 或更高版本。
如果版本低，执行：

pip install --upgrade --extra-index-url https://pypi.nvidia.com "nvidia-physicsnemo[cu12,nn-extras]"
2. 下载 PhysicsNeMo 源码，因为 examples 在 GitHub 里

即使你已经 pip install nvidia-physicsnemo，examples 通常不会完整放在当前工作目录，所以建议 clone 仓库：

cd ~
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo/examples/cfd/external_aerodynamics/transformer_models

确认目录里有这些文件：

ls

应该能看到类似：

README.md  requirements.txt  src

GitHub 页面显示该目录下包含 src、README.md 和 requirements.txt。

3. 安装 GeoTransolver example 的 requirements

你给的 requirements.txt 是：

hydra-core
tabulate
tensorboard
termcolor
torchinfo
einops
transformer_engine[pytorch]
tensorstore
zarr>=3.0

在当前目录执行：

pip install -r requirements.txt

如果 transformer_engine[pytorch] 安装失败，可以单独试：

pip install --extra-index-url https://pypi.nvidia.com "transformer_engine[pytorch]"

然后验证：

python - <<'EOF'
import torch
import zarr
import tensorstore
import einops
import transformer_engine
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("zarr:", zarr.__version__)
print("TransformerEngine import OK")
EOF
4. 先明确：没有 DrivAerML Zarr 数据，不能直接训练

这个 example 使用的是 DrivAerML dataset，并且 README 明确说需要先通过 PhysicsNeMo-Curator 处理成 Zarr 输出文件。
PhysicsNeMo-Curator 的说明也写到，它会读取 CFD 数据，例如 STL、VTU、VTP 和流场变量，然后输出适合训练的 NumPy 或 Zarr 数据。

所以流程是：

DrivAerML 原始数据
        ↓
PhysicsNeMo-Curator 预处理
        ↓
Zarr 格式训练数据
        ↓
GeoTransolver 训练
        ↓
inference_on_zarr.py 推理
5. 下载并处理 DrivAerML 数据

建议新开一个目录放数据：

mkdir -p ~/datasets/drivaerml
mkdir -p ~/datasets/drivaerml_processed_surface

下载 PhysicsNeMo-Curator：

cd ~
git clone https://github.com/NVIDIA/physicsnemo-curator.git
cd physicsnemo-curator

安装 curator：

pip install -e .

进入外部气动 example：

cd examples/external_aerodynamics

下载一小部分数据先测试，例如 run 1 到 run 5：

chmod +x download_hugging_face_dataset.sh
./download_hugging_face_dataset.sh -d ~/datasets/drivaerml -s 1 -e 5

Curator README 说明默认可以下载 1–5 个 run，也可以用 -d、-s、-e 指定下载目录和范围；完整 DrivAerML 有 500 个 run。

6. 用 Curator 生成 surface Zarr 数据

如果你先跑 surface GeoTransolver，执行：

cd ~/physicsnemo-curator

python run_etl.py \
    --config-dir=examples/external_aerodynamics/config \
    --config-name=external_aero_etl_drivaerml \
    etl.source.input_dir=~/datasets/drivaerml \
    etl.sink.output_dir=~/datasets/drivaerml_processed_surface \
    etl.common.model_type=surface

官方 Curator README 给出的 DrivAerML 处理命令就是这种结构：run_etl.py，指定 input_dir、output_dir 和 etl.common.model_type=surface。

如果之后要跑 volume，需要改成：

mkdir -p ~/datasets/drivaerml_processed_volume

python run_etl.py \
    --config-dir=examples/external_aerodynamics/config \
    --config-name=external_aero_etl_drivaerml \
    etl.source.input_dir=~/datasets/drivaerml \
    etl.sink.output_dir=~/datasets/drivaerml_processed_volume \
    etl.common.model_type=volume

但我建议你先跑 surface，因为 surface 数据通常更轻，更适合先验证流程。

7. 回到 GeoTransolver 目录
cd ~/physicsnemo/examples/cfd/external_aerodynamics/transformer_models

先看有哪些配置：

find src/conf -maxdepth 2 -type f

你重点找这两个：

geotransolver_surface
geotransolver_volume

README 明确说明 Hydra 配置在 src/conf 中，可以控制模型、数据、优化器和训练参数。

8. 计算 normalization 文件

训练前需要先计算归一化因子。README 说明 compute_normalizations.py 会生成类似 surface_fields_normalization.npz 的文件，训练时会通过 data.normalization_dir 读取。

建议先这样试：

export HYDRA_FULL_ERROR=1

python src/compute_normalizations.py \
    --config-name geotransolver_surface \
    data.data_dir=~/datasets/drivaerml_processed_surface \
    data.normalization_dir=./normalization

如果报错说 data.data_dir 这个字段不存在，先打开配置文件确认真实字段名：

grep -R "data_dir\|zarr\|normalization" -n src/conf

然后把命令里的 data.data_dir 改成配置文件中实际使用的数据路径字段。因为不同版本的配置字段名可能略有变化，但核心逻辑就是：把训练数据目录指向 Curator 生成的 Zarr 目录。

9. 单 GPU 运行 GeoTransolver Surface 训练

你的 RTX 5880 Ada 有 46GB 显存，先用较小点数测试比较稳。README 提到可以通过 data.resolution=N 做 on-the-fly downsampling，N 表示每张 GPU 使用多少点。

先跑一个小测试：

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name geotransolver_surface \
    data.data_dir=~/datasets/drivaerml_processed_surface \
    data.normalization_dir=./normalization \
    data.resolution=20000 \
    training.num_epochs=2 \
    training.save_interval=1 \
    output_dir=./outputs/geotransolver_surface_test

如果显存足够，再提高：

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name geotransolver_surface \
    data.data_dir=~/datasets/drivaerml_processed_surface \
    data.normalization_dir=./normalization \
    data.resolution=50000 \
    training.num_epochs=100 \
    training.save_interval=10 \
    output_dir=./outputs/geotransolver_surface
10. 如果你遇到 PyTorch 版本 / Muon optimizer 报错

README 里写到，这个训练默认使用 Muon optimizer，而 Muon 只在 pytorch>=2.9.0 中可用。

你的驱动是：

NVIDIA Driver 535.154.05
CUDA 12.2

所以如果当前 PyTorch 不是 2.9，可能会出现类似：

AttributeError: module torch.optim has no attribute Muon

这时有两个方案。

方案 A：改配置用 AdamW

先查配置文件里 optimizer 字段：

grep -R "Muon\|AdamW\|optimizer" -n src/conf

如果能通过 Hydra 覆盖，就类似这样：

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name geotransolver_surface \
    data.data_dir=~/datasets/drivaerml_processed_surface \
    data.normalization_dir=./normalization \
    optimizer.name=adamw \
    training.num_epochs=2 \
    output_dir=./outputs/geotransolver_surface_test

具体字段名要以 src/conf 里实际配置为准。

方案 B：升级 PyTorch，但要小心驱动兼容

如果你强行升级到很新的 PyTorch CUDA runtime，可能需要更高 NVIDIA 驱动。你的 535 驱动比较适合 CUDA 12.2 时代环境。为了稳定，我更建议你先走 AdamW 覆盖方案，确认模型流程跑通后再考虑升级驱动和 PyTorch。

11. 运行推理

训练完成后，假设模型输出目录是：

./outputs/geotransolver_surface

运行：

CUDA_VISIBLE_DEVICES=0 python src/inference_on_zarr.py \
    --config-name geotransolver_surface \
    run_id=./outputs/geotransolver_surface \
    data.data_dir=~/datasets/drivaerml_processed_surface \
    data.normalization_dir=./normalization

README 说明 inference_on_zarr.py 会在验证集 Zarr 数据上推理，并汇总 L2 指标；对于 surface 数据，还会计算 drag/lift coefficient 和 R² 相关性。

你现在最推荐直接执行的版本

你可以从这里开始：

conda activate physicsnemo

cd ~
git clone https://github.com/NVIDIA/physicsnemo.git
cd ~/physicsnemo/examples/cfd/external_aerodynamics/transformer_models

pip install -r requirements.txt

python - <<'EOF'
import torch, physicsnemo, zarr, tensorstore, einops
print("PhysicsNeMo:", physicsnemo.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Basic imports OK")
EOF

然后准备数据：

cd ~
git clone https://github.com/NVIDIA/physicsnemo-curator.git
cd physicsnemo-curator
pip install -e .

cd examples/external_aerodynamics
chmod +x download_hugging_face_dataset.sh
./download_hugging_face_dataset.sh -d ~/datasets/drivaerml -s 1 -e 5

cd ~/physicsnemo-curator
python run_etl.py \
    --config-dir=examples/external_aerodynamics/config \
    --config-name=external_aero_etl_drivaerml \
    etl.source.input_dir=~/datasets/drivaerml \
    etl.sink.output_dir=~/datasets/drivaerml_processed_surface \
    etl.common.model_type=surface

最后训练：

cd ~/physicsnemo/examples/cfd/external_aerodynamics/transformer_models
export HYDRA_FULL_ERROR=1

python src/compute_normalizations.py \
    --config-name geotransolver_surface \
    data.data_dir=~/datasets/drivaerml_processed_surface \
    data.normalization_dir=./normalization

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name geotransolver_surface \
    data.data_dir=~/datasets/drivaerml_processed_surface \
    data.normalization_dir=./normalization \
    data.resolution=20000 \
    training.num_epochs=2 \
    training.save_interval=1 \
    output_dir=./outputs/geotransolver_surface_test

如果这一步能跑出 loss，就说明 GeoTransolver 的环境、数据和 GPU 都已经打通了。
