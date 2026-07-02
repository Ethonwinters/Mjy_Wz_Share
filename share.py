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

over a period of 200 years
just under 略低于
the growth of 
total population 
a comparative breakdown of population proportions


(The provided tables) illustrate (the growth of) New York City’s total population alongside (a comparative breakdown of population proportions) (across five distinct districts) between 1800 and 2000.

It is clear that New York's population (grew at) an extremely rapid rate (over nearly two hundred years), with (the number of citizens) (increasing from 79,216 in 1800 to 8,009,185 in 2000.)  with 是一个介词

Overall, New York City (experienced) (an exponential population explosion) over nearly two hundred years, as the total number of residents (skyrocketed from just under 80,000 to over 8 million).   as 在这里当连词用 while  although  when 
Furthermore, the (demographic distribution) across the various districts (underwent a remarkable change).

In 1800, (fewer than 80,000 citizens) lived in New York. 
By 1900, the population (had increased) (more than forty-fold), reaching 3,437,202. 
By 2000, New York's population size (had grown by) over 4,500,000 (compared to) the 1900 figure.

In 1800, the population of New York (stood at just under 80,000). 
(Over the subsequent century), this figure witnessed an astonishing forty-fold surge, skyrocketing to 3,437,202 by 1900. 
This upward trajectory (persisted into) 2000, by which time the city's population (had expanded by a further 4.5 million).  by a further + 数字：进一步增加了……

Regarding the individual districts, Manhattan (held a dominant position) in 1800, housing 76% of the city’s population (60,515 residents), while the other four districts (combined accounted for only 24%) (18,701).
By 1900, although Manhattan's population (grew significantly to 1,850,093), its share dropped to 54%. (In stark contrast), (the collective population of the other districts) rose to 1,587,109, capturing 46% of the total. 
By 2000, a dramatic reversal occurred as Manhattan's population declined to 1,538,096, with its proportion plummeting to just 19%. 
Conversely, the other districts experienced a massive explosion to 6,471,089, commanding a staggering 81% of the total.(making up a huge 81% of the total.)

  
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



conda create -n physicsnemo_1 python=3.11 -y
conda activate physicsnemo_1

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
personality

pip install --extra-index-url https://pypi.nvidia.com "transformer_engine[pytorch]==2.3.0"



  
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


(physicsnemo_1) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/physicsnemo/examples/cfd/external_aerodynamics/transformer_models$ pip install --extra-index-url https://pypi.nvidia.com "transformer_engine[pytorch]==2.3.0"
Looking in indexes: https://pypi.org/simple, https://pypi.nvidia.com
Collecting transformer_engine==2.3.0 (from transformer_engine[pytorch]==2.3.0)
  Downloading transformer_engine-2.3.0-py3-none-any.whl (486 kB)
Collecting transformer_engine_cu12==2.3.0 (from transformer_engine==2.3.0->transformer_engine[pytorch]==2.3.0)
  Downloading transformer_engine_cu12-2.3.0-py3-none-manylinux_2_28_x86_64.whl (266.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 266.4/266.4 MB 5.0 MB/s  0:00:51
Requirement already satisfied: pydantic in /home/adminstrator/miniforge3/envs/physicsnemo_1/lib/python3.11/site-packages (from transformer_engine_cu12==2.3.0->transformer_engine==2.3.0->transformer_engine[pytorch]==2.3.0) (2.13.4)
Requirement already satisfied: importlib-metadata>=1.0 in /home/adminstrator/miniforge3/envs/physicsnemo_1/lib/python3.11/site-packages (from transformer_engine_cu12==2.3.0->transformer_engine==2.3.0->transformer_engine[pytorch]==2.3.0) (9.0.0)
Requirement already satisfied: packaging in /home/adminstrator/miniforge3/envs/physicsnemo_1/lib/python3.11/site-packages (from transformer_engine_cu12==2.3.0->transformer_engine==2.3.0->transformer_engine[pytorch]==2.3.0) (26.0)
Collecting transformer_engine_torch==2.3.0 (from transformer_engine[pytorch]==2.3.0)
  Downloading transformer_engine_torch-2.3.0.tar.gz (165 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [26 lines of output]
      Traceback (most recent call last):
        File "<string>", line 17, in <module>
      ModuleNotFoundError: No module named 'torch'
      
      The above exception was the direct cause of the following exception:
      
      Traceback (most recent call last):
        File "/home/adminstrator/miniforge3/envs/physicsnemo_1/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
        File "/home/adminstrator/miniforge3/envs/physicsnemo_1/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/adminstrator/miniforge3/envs/physicsnemo_1/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 143, in get_requires_for_build_wheel
          return hook(config_settings)
                 ^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-k4d18br1/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 333, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=[])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-k4d18br1/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 301, in _get_build_requires
          self.run_setup()
        File "/tmp/pip-build-env-k4d18br1/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 520, in run_setup
          super().run_setup(setup_script=setup_script)
        File "/tmp/pip-build-env-k4d18br1/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 19, in <module>
      RuntimeError: This package needs Torch to build.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'transformer_engine_torch' when getting requirements to build wheel


× Failed to build installable wheels for some pyproject.toml based projects
╰─> transformer_engine_torch
26  6
14  4+1+1=6    40 6 
###########################################################################################################################################################################################################################################################################
Invasion of the Robot Umpires
n.侵略，入侵  / ɪnˈveɪʒ(ə)n /n.（体育比赛中的）裁判  / ˈʌmpaɪə(r) /
judgment   n.判断，看法  consensus   / kənˈsensəs /  n.一致看法，共识
conscious of   / ˈkɒnʃəs /  adj.意识到的；
legalistic  / ˌliːɡəˈlɪstɪk /  adj.尊重法律的
legality  / lɪˈɡæləti /n.合法；合法性；墨守法规
机器裁判的入侵

A few years ago, Fred DeJesus from Brooklyn, New York became the first umpire in a minor league baseball game to use something called the Automated Ball-Strike System (ABS), often [referred to as] the 'robo-umpire'. 
「Q27」 Instead of making any judgments himself about a strike*, DeJesus had decisions fed to him through an [earpiece耳机], connected to a modified missile-tracking system. The [contraption奇妙的装置] looked like a large black pizza box with one glowing发光的 green eye;
it was mounted above the press stand.

「Q33」 Major League Baseball (MLB), who had [commissioned] the system, wanted human umpires to announce the calls, just as they would have done in the past. When the first [pitch投球] came in, [a recorded voice录音] told DeJesus it was a strike. 
「Q34」 Previously, calling a strike was a [judgment call主观决策、裁决或意见] on the part of the umpire. 「Q35」 Even if the batter does not hit the ball, a pitch that passes through the 'strike zone' (an [imaginary想象的，虚构的] zone about seventeen inches wide, 
[stretching from 从……延伸] the batter's [knees膝盖] to the middle of his [chest胸部]) is considered a strike. 「Q37」 「Q27」 During that first game, when DeJesus announced calls, there was no heckling and no shouted disagreement. Nobody said a word.

「Q36」 For a hundred and fifty years or so, the strike zone has been the game's [animating force核心动力]—countless arguments between a team's manager and the umpire [have taken place发生] over its boundaries and whether a ball had crossed through it. The rules of play
have evolved in various stages. Today, everyone knows that you may scream your disagreement in an umpire's face, but you must never [shout personal abuse大声谩骂] at them or touch them. That's a no-no. When the robo-umpires came, 「Q37」 however, the arguments stopped.

「Q28」 During the first robo-umpire season, players complained about some strange calls. In response, MLB decided to [tweak扭，拧；稍稍调整] [the dimensions of ...的尺寸]the zone, and the following year the consensus was that ABS is [profoundly consistent高度一致]. MLB says the device is near-perfect,
[precise to within fractions of an inch(precise to within + 某个误差范围 误差不超过……)]. "It'll reduce [controversy (n.争论，争议)] in the game, and be good for the game," says Rob Manfred, who is Commissioner for MLB. But the question is whether controversy [is worth reducing], 
or whether it is the sign of a human hand.

「Q38」 A human, at least, (yells back). When I spoke with Frank Viola, a coach for a North Carolina team, he said that ABS works as designed, but that it was also unforgiving and pedantic, almost legalistic. "Manfred is a lawyer," Viola noted. Some pitchers have 
complained that, compared with a human's, the robot's strike zone seems too precise. Viola was once a major-league player himself. When he was pitching, he explained, umpires rewarded skill. "Throw it where you [aimed n.目标 v.瞄准], and it would be a strike, even if it was an inch or 
two outside. There was a [dialogue / ˈdaɪəlɒɡ /]between pitcher and umpire."

The executive tasked with running the experiment for MLB is Morgan Sword, who's in charge of baseball operations. 「Q39」 According to Sword, ABS was part of a larger project to make baseball more exciting since executives领导层 are terrified of losing younger fans, as has 
been the case with horse racing and boxing. He explains how they began the process by asking fans what version of baseball they found most exciting. The results showed that everyone wanted more action: more hits, more defense, more baserunning. 
「Q30」 This type of baseball essentially hasn't existed since the 1960s, when the hundred-mile-an-hour fastball, which is difficult to hit and control, entered the game. It flattened the game into strikeouts, walks, and home runs—[a type of] play lacking much action.
be tasked with doing something 被指派去做某事 被委派负责运行这项实验    be in charge of+n 负责....

Sword's team brainstormed [potential fixes潜在的解决方案]. Any rule that existed, they talked about changing—from changing the [bats球棒] to changing the geometry of the field. But while all of these were [ruled out排除在外] as potential fixes, ABS was seen as a [perfect vehicle 完美载体]for change. 
According to Sword, once you get the [technology/ tekˈnɒlədʒi /] right, you can load any strike zone you want into the system. 「Q32」 "It might be a triangle, or a blob, or something shaped like Texas. Over time, as baseball evolves, ABS can allow the zone to change with it."
rule out of  排除在外  be seen as 被视为

"In the past twenty years, sports [have moved away from] judgment calls. Soccer has Video Assistant Referees (for offside decisions, for example). Tennis has Hawk-Eye (for line calls, for example). For almost a decade, baseball has used instant replay on the base paths.
This is widely liked, even if the precision can sometimes cause problems. But these applications deal with something physical: bases, lines, goals. The boundaries of action are precise, delineated like the keys of a piano. This is not the case with ABS and the strike zone.
Historically, a certain discretion [has been appreciated.一直受到赞赏]
delineate v./ dɪˈlɪnieɪt /（详细地）描述，解释；标明，标示（边界）  discretion / dɪˈskreʃn /  n.自行决定权，判断力；


I decided to email Alva Noë;, a professor at Berkeley University and a baseball fan, for his opinion. "Hardly a day goes by that I don't wake up and run through the reasons that this [robo-umpires] is such a terrible idea," he replied. He later told me, "This is
part of a movement to use algorithms to take the hard choices of living out of life." Perhaps he's right. We watch baseball to kill time, not to maximize it. Some players I have met take a dissenting stance toward the robots too, believing that accuracy is not the answer.
According to Joe Russo, who plays for a New Jersey team, "With technology, people just want everything to be perfect. That's not reality. I think perfect would be weird. Your teams are always winning, work is always just great, there's always money in your pocket, your car
never breaks down. What is there to talk about?"
dissent   / dɪˈsent /v.n.持异议，不同意  stance n.（公开表明的）观点，态度 ,立场  reality / riˈæləti /n.真实，现实
Hardly a day goes by.   几乎每天都……  
Hardly a day goes by that I don't wake up... 几乎没有一天我醒来时不……    run through something   仔细回想
This is part of a movement   这是某种运动（趋势）的一部分。    take ... out of ...   把……从……中移除  

* strike: a strike is when the batter swings at a ball and misses or when the batter does not swing at a ball that passes through the strike zone.

yes no not given
27When DeJesus first used ABS, he [shared decision-making 共同决策] about strikes with it.   A   B
 
28MLB considered it necessary to [amendv.修改，修订] the size of the strike zone when criticisms were received from players.C  A

29MLB is keen to justify the money spent on improving the accuracy of ABS's calculations.C
 
30The hundred-mile-an-hour fastball led to a more exciting style of play. A  B

31The differing proposals for [alterations] to the baseball bat led to [fierce adj.（动作或情感）强烈的，激烈的] [debate n.v. 讨论，辩论] on Sword's team.   C

32ABS makes changes to the shape of the strike zone feasible. A


Even after ABS was developed, MLB still wanted human umpires to shout out decisions as they had in their 33. F
The umpire's job had, at one time, required a 34D about whether a ball was a strike. A ball is considered a strike when the batter does not hit it and it crosses through a 35 H extending approximately from the batter's knee to his chest.
In the past, 36 B over strike calls were not uncommon, but today everyone accepts the complete ban on pushing or shoving the umpire. 
One difference, however, is that during the first game DeJesus used ABS, strike calls were met with 37G. 

A pitch boundary  #
B nemerous disputes 大量争议
C team tactics
D subject assessment  主观评估 
E widespread approval
F former roles # 以前的职责
G total silence
H percieved area # 人眼判断的区域

38What does the writer suggest about ABS in the fifth paragraph? C B
 It [is bound to 必然]make key decisions that are wrong.
 It may reduce some of the [appeal 吸引力] of the game.
 It will lead to the [disappearance 消失] of human umpires.
 It may increase calls for the rules of baseball to be changed.
 
39Morgan Sword says that the introduction of ABS D
 was regarded as an experiment without a guaranteed outcome.
 was intended to keep up with developments in other sports.
 was a response to changing attitudes about the role of sport.
 was an attempt to ensure baseball retained a young audience.
 
40Why does the writer include the views of Noë and Russo?  C
 to show that attitudes to technology [vary widely 差异很大]
 to argue that people [have unrealistic expectations of sport]   realistic  / ˌriːəˈlɪstɪk / adj.务实的，实事求是的；实际的
 to indicate that accuracy is not the same thing as enjoyment
 to suggest that the number of baseball fans needs to increase
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
Procrastination 英 / prəˌkræstɪˈneɪʃ(ə)n /拖延
procrastinate v.拖延，耽搁
chronic  / ˈkrɒnɪk / adj.（疾病）慢性的，长期的；（问题）一直有的，反复出现的；   chronic disease  慢性病
esteem  / ɪˈstiːm / n.v.尊重，敬重  self-esteem 自尊心
harsh  / hɑːʃ /  adj.（环境）恶劣的，艰苦的；严厉的，残酷的；刺耳的，难听的；
plagiarism  / ˈpleɪdʒərɪzəm / n. 剽窃，抄袭；剽窃作品  plagiarize  v.
fraudulent   / ˈfrɔːdʒələnt /adj.欺诈的，诈骗的
extension  / ɪkˈstenʃ(ə)n / n.延伸，扩展；展期  extensive  adj. 广阔的；广泛的；巨大的

psychologist  / saɪˈkɒlədʒɪst / 
physiologist  / ˌfɪziˈɒlədʒɪst /

A psychologist explains why we [put off推迟] important tasks and how we can [break this habit]

A  Procrastination is the habit of delaying a necessary task, usually by focusing on less urgent, more enjoyable, and easier activities instead. We all do it [from time to time时不时地] We might be [composing a message编辑一条信息] to a friend who we have to [let down],
or [putting together] an important report for college or work; we're doing our best to avoid doing the job at hand, but [deep down实际上；在心底] we know that we should just be getting on with it. 
Unfortunately, [berating / bɪˈreɪt/ ourselves 自我苛责] won't stop us procrastinating again. In fact, it's one of the [worst / wɜːst /] things we can do. This matters because, as my research shows,procrastination doesn't just waste time, but is actually linked to other problems, too.

B    「Q17」 「Q14」 [Contrary / ˈkɒntrəri / to] popular belief, procrastination is not due to laziness or poor time management. 「Q16」 Scientific studies suggest procrastination is, in fact, caused by poor mood management. [This makes sense] if we consider 
that people are more likely to put off starting or completing tasks that they are really not keen to do. 
「Q18」 If just thinking about the task [threatens(v) our sense of self-worth] or makes us anxious, we will be more likely to put it off. 「Q19」 「Q16」 「Q16」 Research involving brain imaging has found that areas of the brain linked to detection of
threats and emotion regulation are actually different in people who chronically procrastinate compared to those who don't procrastinate frequently.

C    「Q20」 Tasks that are emotionally loaded or difficult, such as preparing for exams, are [prime candidates for 首要原因procrastination]. 
「Q21」 People with low self-esteem are more likely to procrastinate. Another group of people who tend to procrastinate are perfectionists, who worry their work will be judged harshly by others. We know that if we don't
finish that report or complete those home repairs, then what we did can't be evaluated. 
When we avoid such tasks, we also avoid the negative emotions associated with them. This is rewarding, and it [conditions v.训练，使习惯于] us to use procrastination to repair our mood.
If we engage in more enjoyable tasks instead, we get another mood boost. [In the long run长远看来], however, 
procrastination isn't an effective way of managing emotions. 「Q22」 The 'mood repair' we experience is temporary. [Afterwards事后], people tend to be left with a sense of guilt that not only increases their negative mood, but also [reinforces强化 their tendency (n) to procrastinate.]

D    So why is this such a problem? When most people think of the costs of procrastination, they think of [the toll损害、代价、负面影响 on productivity]. For example, studies have shown that procrastination negatively impacts on student performance. But putting off reading textbooks
and writing essays may affect other areas of students' lives. In one study of over 3,000 
German students [over a six-month period], those who reported procrastinating over their university work were also [more likely to engage in study-related / rɪˈleɪtɪd / misconduct<正式>不端行为], such as cheating and plagiarism. But the behaviour that procrastination was most closely linked with
was using fraudulent excuses to get deadline extensions. 
Other research shows that employees on average spend almost a quarter of their workday procrastinating, and again this is linked with negative outcomes. In fact, 「Q23-24」 in one US survey of over 22,000 employees, participants who said they regularly procrastinated had less annual income 
and [less employment / ɪmˈplɔɪmənt / stability/ stəˈbɪləti /]. For every one-point increase on a measure of chronic procrastination, annual income [decreased by US$15,000. 减少了]

E    Procrastination also correlates with serious health and well-being problems. A tendency to procrastinate is linked to poor mental health, including higher levels of [depression n. and anxiety n.抑郁和焦虑]. Across numerous studies, I've found people who regularly procrastinate
report a greater number of health issues, such as [headache]s 头痛, flu流感 and colds, and [digestive / daɪˈdʒestɪv /issues]. 
They also experience higher levels of stress and poor sleep quality. They are [less likely to更不容易] practise healthy behaviours, such as eating a healthy diet and regularly exercising, and use [destructive/ dɪˈstrʌktɪv /adj.破坏性的，毁灭性的 (coping strategies)应对策略]
to manage their stress. In one study of over 700 people, I found people (prone to)倾向于 procrastination had a 63% greater risk of poor heart health after [accounting for other personality traits 个性特征and demographics adj.人口的，人口统计的 n.（具有某种特征的）群体；人口统计数据.]

F     [Finding better ways of managing our emotions is one route路线 途径 out of the vicious cycle恶性循环 of procrastination]. An important first step is to manage our environment and how we view the task. 
There are a number of evidence-based strategies 基于实证的策略that can 「Q25-26」 help us fend v.避开，挡开 off 抵挡 distractions n.使人分心的事物，干扰 that can occupy our minds 占据我们的头脑when we should be focusing on the thing we should be getting on with. 
For example, reminding ourselves about why the task is important and valuable can increase positive feelings towards it. Forgiving (forgive v.原谅，宽恕)ourselves and feeling compassion(n.同情，怜悯) when we procrastinate can help break the procrastination cycle. 
We should admit 坦然接受that we feel bad, but not be [overly critical of ourselves]. 
We should remind ourselves that we're not the first person to procrastinate, nor the last. Doing this can (take the edge off) 减轻，缓解the negative feelings we have about ourselves when we procrastinate. This can all make it easier to (get back on track)重新回到正轨.
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
A-F
14 mention of false assumptions about why people procrastinate  B
15 reference to the realisation that others also procrastinate C  F
16 neurological evidence of a link between procrastination and emotion B

Many people think that procrastination is the result of 17(lazines). Others believe it to be the result of an inability to organise time efficiently.

But scientific studies suggest that procrastination is actually due to poor mood management. The tasks we are most likely to put off are those that could damage our self-esteem or cause us to feel 18(anxious) when we think about them. 
Research comparing chronic procrastinators with other people even found differences in the brain regions associated with regulating emotions and identifying 19(threats).

Emotionally loaded and difficult tasks often cause us to procrastinate. Getting ready to take 20 (exams) might be a typical example of one such task. People who are likely to procrastinate tend to be either 21(perfectionists)
or those with low self-esteem.

Procrastination is only a short-term measure for managing emotions. It's often followed by a feeling of guilt22 ,which worsens our mood and leads to more procrastination.


23-24Which TWO comparisons between employees who often procrastinate and those who do not are mentioned in the text?  AC
 A. Their salaries are lower. 1
 B. The quality of their work is inferior.1 / ɪnˈfɪəriə(r) /  adj.次的，较差的；低等的，下级的
 C. They don’t keep their jobs for as long. 1
 D. They don’t enjoy their working lives as much.
 E. They have poorer relationships with colleagues.


25-26Which TWO recommendations for getting out of a cycle of procrastination does the writer give? D E   AE
 A. not judging ourselves harshly  不要苛责自己
 B. setting ourselves manageable aims
 C. rewarding ourselves for tasks achieved  x
 D. prioritising tasks according to their importance
 E. avoiding things that stop us concentrating on our tasks

###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
Manatees
limb / lɪm /n.肢  flipper n. 鳍 tail / teɪl /  n.（动物的）尾巴


Manatees, also known as sea cows, are [aquatic mammals] that belong to a group of animals called Sirenia. This group also contains dugongs. Dugongs and manatees [look quite alike] - they are similar in size, colour and shape, and both have flexible flippers for forelimbs. 
「Q1」 However, the manatee has a broad, rounded tail, whereas the dugong's is fluked, like that of a whale. There are three species of manatees: the West Indian manatee (Trichechus manatus), the African manatee (Trichechus senegalensis) and the Amazonian manatee (Trichechus inunguis).

Unlike most mammals, manatees have only six bones in their neck - most others, including humans and giraffes, have seven. This short neck allows a manatee to move its head up and down, but not side to side. 「Q2」 To see something on its left or its right, a manatee must turn its entire body, 
[steering with] its flippers. Manatees have pectoral flippers but no back limbs, only a tail for propulsion. They do have pelvic bones, however - a [leftover] from their evolution from a four-legged to a fully aquatic animal. Manatees share some visual similarities to elephants. 
Like elephants,manatees have [thick, wrinkled skin]. 「Q3」 They also have some hairs covering their bodies which help them sense vibrations in the water around them.

「Q4」 Seagrasses and other [marine plants] [make up] [most of a manatee's diet]. Manatees spend about eight hours each day grazing and uprooting plants. They eat up to 15% of their weight in food each day. African manatees are omnivorous - 
studies have shown that [mollusc]s and fish make up a small part of their diets. West Indian and Amazonian manatees are both [herbivore]s.

Manatees' teeth are all [molar]s - flat, rounded teeth for [grind]ing food. Due to manatees' abrasive aquatic plant diet, these teeth [get worn down] and they eventually [fall out], so they continually grow new teeth that get pushed forward to replace the ones they lose.
「Q5」 Instead of having [incisor]s to grasp their food, manatees have [lip]s which function like a pair of hands to help tear food away from the seafloor.

Manatees are fully aquatic, but as mammals, they need to come up to the surface to breathe. When [awake], they typically surface every two to four minutes, but they can hold their breath for much longer. 
Adult manatees sleep underwater for 10-12 hours a day, but they come up for air every 15-20 minutes. Active manatees need to breathe more frequently. 
「Q6」 It's thought that manatees use their muscular diaphragm and breathing to adjust their buoyancy. They may use diaphragm [contractions] to compress and store gas in folds in their large [intestine] to help them float.

The West Indian manatee reaches about 3.5 metres long and weighs on average around 500 kilogrammes. 
「Q7」 It moves between fresh water and salt water, taking advantage of coastal mangroves and coral reefs, rivers, lakes and inland lagoons. There are two subspecies of West Indian manatee: the Antillean manatee is found in waters from the Bahamas to Brazil,
whereas the Florida manatee is found in US waters, although some individuals have been recorded in the Bahamas. 
「Q8」 In winter, the Florida manatee is typically restricted to Florida. When the ambient water temperature drops below 20℃, it [takes refuge in] naturally and artificially warmed water, such as at the warm-water outfalls from powerplants.

「Q9」 The African manatee is also about 3.5 metres long and found in the sea along the west coast of Africa, from Mauritania down to Angola. The species also makes use of rivers, with the mammals seen in landlocked countries such as Mali and Niger.

The Amazonian manatee is the smallest species, though it is still a big animal. It grows to about 2.5 metres long and 350 kilogrammes.
Amazonian manatees favour calm, shallow waters that are above 23℃. This species is found in fresh water in the Amazon Basin in Brazil, as well as in Colombia, Ecuador and Peru.

All three manatee species are endangered or at a heightened risk of [extinction]. The African manatee and Amazonian manatee are both listed as Vulnerable by the International Union for Conservation of Nature (IUCN). 
「Q10」 It is estimated that 140,000 Amazonian manatees were killed between 1935 and 1954 for their meat, fat and skin, with the latter used to make leather. 
In more recent years, African manatee decline has been tied to [incidental capture偶然捕获] in fishing nets and hunting. Manatee hunting is now [illegal] in every country the African species is found in.

「Q11」 The two subspecies of West Indian manatee are listed as Endangered by the IUCN. Both are also expected to undergo a decline of 20% over the next 40 years. 
「Q12」 A review of almost 1,800 cases of [entanglement] in fishing nets and of [plastic consumption] among marine mammals in US waters from 2009 to 2020 found that at least 700 cases involved manatees. 
「Q13」 The chief cause of death in Florida manatees is [boat strikes]. However, laws in certain parts of Florida now limit boat speeds during winter, allowing slow-moving manatees more time to respond.

 Appearance

 • look similar to dugongs, but with a differently shaped 1tail
 
 Movement

 • have fewer neck bones than most mammals

 • need to use their 2 flippers to help to turn their bodies around in order to look sideways

 • sense vibrations in the water by means of 3hairs on their skin
by means of  通过，借助于：用某种方法或手段来实现某事

 Feeding

 • eat mainly aquatic vegetation, such as 4.Seagrasses
水生植物
 • grasp and pull up plants with their 5lips.
嘴唇

 Breathing

 • come to the surface for air every 2-4 minutes when awake and every 15-20 while sleeping
adj.醒着的
 • may regulate the 6buoyancy of their bodies by using muscles of diaphragm to store air internally
它们可能利用横膈膜（diaphragm）的肌肉在体内储存空气，从而调节身体的 buoyancy（浮力）

7West Indian manatees can be found in a variety of different aquatic habitats. A
 TRUE
 FALSE
 NOT GIVEN
8The Florida manatee lives in warmer waters than the Antillean manatee. C
 TRUE
 FALSE
 NOT GIVEN
9The African manatee's range is limited to coastal waters between the West African countries of Mauritania and Angola. A B
 TRUE
 FALSE
 NOT GIVEN
10The extent of the loss of Amazonian manatees in the mid-twentieth century was only revealed many years later.C
 TRUE
 FALSE
 NOT GIVEN
11It is predicted that West Indian manatee populations will fall in the coming decades. A
 TRUE
 FALSE
 NOT GIVEN
12The risk to manatees from entanglement and plastic consumption increased significantly in the period 2009-2020. A C
在 2009 至 2020 年期间，海牛因被渔网等缠绕（entanglement）以及误食塑料（plastic consumption）而面临的风险显著增加。
 TRUE
 FALSE
 NOT GIVEN
13There is some [legislation / ˌledʒɪsˈleɪʃn / in place] which aims to reduce the likelihood of boat strikes on manatees in Florida. A
目前已经有一些法律法规出台，旨在降低佛罗里达州海牛遭受船只撞击（boat strikes）的可能性。
 TRUE
 FALSE
 NOT GIVEN
###########################################################################################################################################################################################################################################################################
2+3+4
###########################################################################################################################################################################################################################################################################
Frozen Food

A US perspective on the development of the frozen food industry

At some point in history, humans discovered that ice preserved food. There is evidence that winter ice was stored to preserve food in the summer as far back as 10,000 years ago. Two thousand years ago, the inhabitants of South America's Andean mountains 
had a unique means of conserving potatoes for later consumption. They froze them overnight, then trampled them to squeeze out the moisture, then dried them in the sun. This preserved their nutritional value—if not their aesthetic appeal.

Natural ice remained the main form of refrigeration until late in the 19th century. In the early 1800s, ship owners from Boston, USA, had enormous blocks of Arctic ice towed all over the Atlantic for the purpose of food preservation. In 1851, railroads 
first began putting blocks of ice in insulated rail cars to send butter from Ogdensburg, New York, to Boston.

Finally, in 1870, Australian inventors found a way to make 'mechanical ice'. They used a compressor to force a gas—ammonia at first and later Freon—through a condenser. The compressed gas gave up some of its heat as it moved through the condenser. Then the 
gas was released quickly into a low-pressure evaporator coil where it became liquid and cold. Air was blown over the evaporator coil and then this cooled air passed into an insulated compartment, lowering its temperature to freezing point.

Initially, this process was invented to keep Australian beer cool even in hot weather. But Australian cattlemen were quick to realize that, if they could put this new invention on a ship, they could export meat across the oceans. In 1880, a shipment of Australian 
beef and mutton was sent, frozen, to England. While the food frozen this way was still palatable, there was some deterioration. During the freezing process, crystals formed within the cells of the food, and when the ice expanded and the cells burst, this spoilt the 
flavor and texture of the food.

The modern frozen food industry began with the indigenous Inuit people of Canada. In 1912, a biology student in Massachusetts, USA, named Clarence Birdseye, ran out of money and went to Labrador in Canada to trap and trade furs. While he was there, he became fascinated
with how the Inuit would quickly freeze fish in the Arctic air. The fish looked and tasted fresh even months later.

Birdseye returned to the USA in 1917 and began developing mechanical freezers capable of quick-freezing food. Birdseye methodically kept inventing better freezers and gradually built a business selling frozen fish from Gloucester, Massachusetts. In 1929, his business was 
sold and became General Foods, but he stayed with the company as director of research, and his division continued to innovate.

Birdseye was responsible for several key innovations that made the frozen food industry possible. He developed quick-freezing techniques that reduced the damage that crystals caused, as well as the technique of freezing the product in the package it was to be sold in. He
also introduced the use of cellophane, the first transparent material for food packaging, which allowed consumers to see the quality of the product. Birdseye products also came in convenient size packages that could be prepared with a minimum of effort.

But there were still obstacles. In the 1930s, few grocery stores could afford to buy freezers for a market that wasn't established yet. So, Birdseye leased inexpensive freezer cases to them. He also leased insulated railroad cars so that he could ship his products nationwide. 
However, few consumers had freezers large enough or efficient enough to take advantage of the products.

Sales increased in the early 1940s, when World War II gave a boost to the frozen food industry because tin was being used for munitions. Canned foods were rationed to save tin for the war effort, while frozen foods were abundant and cheap. Finally, by the 1950s, refrigerator 
technology had developed far enough to make these appliances affordable for the average family. By 1953, 33 million US families owned a refrigerator, and manufacturers were gradually increasing the size of the freezer compartments in them.

1950s families were also looking for convenience at mealtimes, so the moment was right for the arrival of the 'TV Dinner'. Swanson Foods was a large, nationally recognized producer of canned and frozen poultry. In 1954, the company adapted some of Birdseye's freezing techniques, 
and with the help of a clever name and a huge advertising budget, it launched the first 'TV Dinner'. This consisted of frozen turkey, potatoes and vegetables served in the same segmented aluminum tray that was used by airlines. The product was an instant success. Within a year, 
Swanson had sold 13 million TV dinners. American consumers couldn't resist the combination of a trusted brand name, a single-serving package and the convenience of a meal that could be ready after only 25 minutes in a hot oven. By 1959, Americans were spending $2.7 billion annually 
on frozen foods, and half a billion of that was spent on ready-prepared meals such as the TV Dinner.

Today, the US frozen food industry has a turnover of over $67 billion annually, with $26.6 billion of that sold to consumers for home consumption. The remaining $40 billion in frozen food sales come through restaurants, cafeterias, hospitals and schools, and that represents a 
third of the total food service sales.

Choose ONE WORD ONLY from the passage for each answer.
 The history of frozen food

 2,000 years ago, South America

 ▪ People conserved the nutritional value of 1potatoes , using a method of freezing then drying.      

 1851, USA

 ▪ 2butter was kept cool by ice during transportation in specially adapted trains. 

 1880, Australia

 ▪ Two kinds of 3 meat  were the first frozen food shipped to England.

  1917 onwards, USA

 ▪ Clarence Birdseye introduced innovations including:

   - quick-freezing methods, so that 4crystals did not spoil the food.

   - packaging products with 5cellophane , so the product was visible.

  Early 1940s, USA

 ▪ Frozen food became popular because of a shortage of 6tin.

  1950s, USA

 ▪ A large number of homes now had a 7. refrigerator

T F N
8The ice transportation business made some Boston ship owners very wealthy in the early 1800s. C

9A disadvantage of the freezing process invented in Australia was that it affected the taste of food. A

10Clarence Birdseye travelled to Labrador in order to learn how the Inuit people froze fish. A   B

11Swanson Foods invested a great deal of money in the promotion of the TV Dinner.A

12Swanson Foods developed a new style of container for the launch of the TV Dinner.C B

13The US frozen food industry is currently the largest in the world.C
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
Can the planet’s coral reefs be saved?

 A    Conservationists have (put the final touches to) a giant artificial reef they have been assembling at the world-renowned Zoological Society of London (London Zoo). Samples of the planet’s most spectacular corals – vivid green branching coral, 
yellow scroll, blue ridge and many more species – have been added to the giant tank along with fish that thrive in their presence: blue tang, clownfish and many others. The reef is in the zoo’s new gallery, Tiny Giants, which is dedicated to the minuscule 
invertebrate creatures that sustain life across the planet. The coral reef tank and its seven-metre-wide window form the core of the exhibition.

‘Coral reefs are the most diverse ecosystems on Earth and we want to show people how wonderful they are,’ said Paul Pearce-Kelly, senior curator of invertebrates and fish at the Zoological Society of London. ‘However, we also want to highlight the research and 
conservation efforts that are now being carried out to try to save them from the threat of global warming.’ They want people to see what is being done to try to save these wonders.

 B    Corals are (composed of) tiny animals, known as polyps, with tentacles for capturing small marine creatures in the sea water. These polyps are transparent but get their brilliant tones of pink, orange, blue, green, etc. from algae that live within them, which in
turn get protection, while their photosynthesising of the sun’s rays provides nutrients for the polyps. This comfortable symbiotic relationship has led to the growth of coral reefs that cover 0.1% of the planet’s ocean bed while providing homes for more than 25% of
marine species, including fish, molluscs, sponges and shellfish.

 C    As a result, coral reefs are often described as the ‘rainforests of the sea’, though the comparison is dismissed by some naturalists, including David Attenborough. ‘People say you cannot beat the rainforest,’ Attenborough has stated. ‘But that is simply not true. 
You go there and the first thing you think is: where … are the birds? Where are the animals? They are hiding in the trees, of course. No, if you want beauty and wildlife, you want a coral reef. Put on a mask and stick your head under the water. The sight is mind-blowing.’

 D    Unfortunately, these majestic sights are now under very serious threat, with the most immediate problem coming in the form of thermal stress. Rising ocean temperatures are triggering bleaching events that strip reefs of their colour and eventually kill them. And that 
is just the start. Other menaces include ocean acidification, sea level increase, pollution by humans, deoxygenation and ocean current changes, while the climate crisis is also increasing habitat destruction. As a result, vast areas – including massive chunks of Australia’s 
Great Barrier Reef – have already been destroyed, and scientists advise that more than 90% of reefs could be lost by 2050 unless urgent action is taken to tackle global heating and greenhouse gas emissions.

Pearce-Kelly says that coral reefs have to survive really harsh conditions – wave erosion and other factors. And ‘when things start to go wrong in the oceans, then corals will be the first to react. And that is exactly what we are seeing now. Coral reefs are dying and they are 
telling us that all is not well with our planet.’

 E    However, scientists are trying to pinpoint hardy types of coral that could survive our overheated oceans, and some of this research will be carried out at London Zoo. ‘Behind our … coral reef tank we have built laboratories where scientists will be studying coral species,’ 
said Pearce-Kelly. One aim will be to carry out research on species to find those that can survive best in warm, acidic waters. Another will be to try to increase coral breeding rates. ‘Coral spawn just once a year,’ he added. ‘However, aquarium-based research has enabled some 
corals to spawn artificially, which can assist coral reef restoration efforts. And if this can be extended for all species, we could consider the launching of coral-spawning programmes several times a year. That would be a big help in restoring blighted reefs.’

 F    Research in these fields is being conducted in laboratories around the world, with the London Zoo centre linked to this global network. Studies carried out in one centre can then be tested in others. The resulting young coral can then be displayed in the tank in Tiny Giants. 
‘The crucial point is that the progress we make in making coral better able to survive in a warming world can be shown to the public and encourage them to believe that we can do something to save the planet’s reefs,’ said Pearce-Kelly. ‘Saving our coral reefs is now a critically 
important ecological goal.’

V II IV VII III VI
Questions 14-19

Reading Passage 2 has six sections, A-F.

Choose the correct heading for each section from the list of headings below.

Write the correct number, i-vii, in boxes 14-19 on your answer sheet.

I. Tried and tested solutions  经过检验并证明有效的解决方案

II. Cooperation beneath the waves  海底的共生合作 

III. Working to lessen the problems  努力减轻这些问题

IV. Disagreement about the accuracy of a certain phrase  关于某个说法是否准确的分歧

V. Two clear educational goals  两个明确的教育目标

VI. Promoting hope  激发希望

VII. A warning of further trouble ahead 对未来更多问题的警告

20-21Which TWO of these causes of damage to coral reefs are mentioned by the writer of the text?  CE
 A. a rising number of extreme storms A   极端风暴数量不断增加
 B. the removal of too many fish from the sea  从海洋中过度捕捞鱼类
 C. the contamination of the sea from waste C  废弃物造成的海洋污染
 D. increased disease among marine species   海洋物种疾病增加
 E. alterations in the usual flow of water in the seas  海洋中通常水流方式的变化

22-23Which TWO of the following statements are true of the researchers at London Zoo? BD
 A. They are hoping to expand the numbers of different corals being bred in laboratories.  他们希望增加在实验室中培育的不同珊瑚种类的数量。
 B. They want to identify corals that can cope well with the changed sea conditions.  他们想找出能够很好适应海洋环境变化的珊瑚
 C. They are looking at ways of creating artificial reefs that corals could grow on.  他们正在研究如何制造可供珊瑚生长的人工礁。
 D. They are trying out methods that would speed up reproduction in some corals.  他们正在尝试能够加快某些珊瑚繁殖速度的方法。
 E. They are investigating materials that might protect reefs from higher temperatures.  他们正在研究可能保护珊瑚礁免受高温影响的材料。

24 Corals have a number of 24tentacles which they use to collect their food
25 Algae gain 25protection from being inside the coral.
26 Increases in the warmth of the sea water can remove the 26  colour  from coral.
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
Robots and us

Three leaders in their fields answer questions about our relationships with robot

When asked 'Should robots be used to (colonise) other planets?', cosmology and astrophysics Professor 😊Martin Rees said he believed the (solar) system would be mapped by robotic craft by the end of the century. 'The next step would be mining of (asteroid)s, 
enabling (fabrication) of large structures in space without having to bring all the raw materials from Earth. I think this is more realistic and benign than the "terraforming"* of planets.' He maintains that colonised planets 'should be preserved with a 
status that is (analogous) to (Antarctica) here on Earth.'

On the question of using robots to colonise other planets and exploit(exploitation) (mineral) resources, engineering Professor 😊Daniel Wolpert replied, 'I don't see a pressing need to colonise other planets unless we can bring [these] resources back to Earth. The vast 
majority of Earth is currently inaccessible to us. Using robots to gather resources nearer to home would seem to be a better use of our robotic tools.'

Meanwhile, for (anthropology) Professor 😊Kathleen Richardson, the idea of 'colonisation' of other planets seemed morally (dubious): 'I think whether we do something on Earth or on Mars we should always do it in the spirit of a genuine interest in "the Other", 
not to impose a particular model, but to meet "the Other".'

In response to the second question, 'How soon will machine intelligence (outstrip) human intelligence?', 😊Rees mentions robots that are advanced enough to beat humans at chess, but then goes on to say, 'Robots are still limited in their ability to sense their 
environment: they can't yet recognise and move the pieces on a real chessboard as cleverly as a child can. Later this century, however, their more advanced successors may relate to their surroundings, and to people, as (adeptly) as we do. Moral questions then 
arise. Should we feel (guilty) about exploiting [(sophisticated) robots]? Should we (fret) if they are (underemployed), frustrated, or bored?'

😊Wolpert's response to the question about machine intelligence outstripping human intelligence was this: 'In a limited sense it already has. Machines can already navigate, remember and search for items with an ability that (far outstrips humans). However, there
is no machine that can identify visual objects or speech with the reliability and flexibility of humans. Expecting a machine close to the creative intelligence of a human within the next 50 years would be highly ambitious.'

😊Richardson believes that our fear of machines becoming too advanced (has more to do with) human nature than anything (intrinsic) to the machines themselves. In her view, it (stems from源于) humans' tendency to （personify inanimate objects）: we create machines based on
representations of ourselves, imagine that machines think and behave as we do, and therefore see them as an autonomous threat. 'One of the consequences of thinking that the problem lies with machines is that we tend to imagine they are greater and more powerful 
than they really are and subsequently they become so.'

This led on to the third question, 'Should we be scared by advances in artificial intelligence?' To this question,😊 Rees replied, 'Those who should be worried are the futurologists who believe in the so-called "singularity".**  And another worry is that we are 
increasingly dependent on computer networks, and that these could behave like a single "brain" with a mind of its own, and with goals that may be (contrary to human welfare). I think we should ensure that robots remain as [no more than] "idiot savants" lacking the 
capacity to outwit us, even though they may greatly surpass us in the ability to calculate and process information.'

😊Wolpert's response was to say that we have already seen the damaging effects of artificial intelligence in the form of computer (virus)es. 'But in this case,' he says, 'the real intelligence is the (malicious) designer. Critically, the benefits of computers outweigh
the damage that computer viruses cause. Similarly, while there may be misuses of robotics in the near future, the benefits that they will bring are likely to outweigh these negative aspects.'

😊Richardson's response to this question was this: 'We need to ask why fears of artificial intelligence and robots (persist); none have in fact risen up and challenged human (supremacy).' She believes that as robots have never shown themselves to be a threat to humans, 
(it seems unlikely that 看起来不太可能…… / 似乎不会……)they ever will. In fact, she went on, 'Not all fear [robots]; many people welcome machine intelligence.'

In answer to the fourth question, 'What can science (fiction) tell us about robotics?', 😊Rees replied, 'I sometimes advise students that it's better to read first-rate science fiction than second-rate science - more (stimulating), 
and perhaps (no more likely 不太可能) to be wrong.'

As his response, 😊Wolpert commented, 'Science fiction has often been remarkable at predicting the future. ...Science fiction has painted a vivid spectrum of possible futures, from cute and helpful robots to dystopian robotic societies. Interestingly, almost no science 
fiction (envisage)s a future without robots.'

Finally, on the question of science fiction, 😊Richardson pointed out that in modern society, people tend to think there is reality on the one hand, and fiction and fantasy on the other. She then explained that the division did not always exist, and that scientists 
and technologists made this separation because they wanted to carve out the sphere of their work. 'But the divide is not so clear cut, and that is why the worlds seem to collide at times,' she said. 'In some cases, we need to bring these different understandings 
together to get a whole perspective. Perhaps then, we won't be so (frightened) that something we create as a copy of ourselves will be a [threat] to us.'

* terraforming: modifying a planet's atmosphere to suit human needs

** singularity:   the point when robots will be able to start creating ever more sophisticated versions of themselves         

less serious 没那么严重
Look at the following statements (Questions 27-33) and the list of experts below.

Match each statement with the correct expert, A, B or C.

NB You may use any letter more than once.

27. For our own safety, humans will need to restrict the abilities of robots.  C A

28. The risk of robots harming us is less serious than humans believe it to be.  B C

29. It will take many decades for robot intelligence to be as (imaginative) as human intelligence.  B

30. We may have to start considering whether we are treating robots fairly.   A

31. Robots are probably of more help to us on Earth than in space.  B

32. The ideas in high-quality science fiction may prove to be just as accurate as those found in the work of (mediocre) scientists.  A

33. There are those who (look forward to) robots developing greater intelligence. C

A Martin Ress
B Daniel Wolpert
C Kathleen Richardson

Questions 34-36

Complete each sentence with the correct ending, A-D, below.

34. Richardson and Rees express similar views regarding the (ethical) aspect of   A C

35. Rees and Wolpert share an opinion about the extent of  B

36. Wolpert disagrees with Richardson on the question of  D

A. robots to explore outer space.

B. advances made in machine intelligence so far.

C. changes made to other planets for our own benefit.

D. the harm already done by artificial intelligence.

37What point does Richardson make about fear of machines? B
 It has grown alongside the development of ever more advanced robots.  ever + 比较级等于 more and more=increasingly  
 It is the result of our (inclination) to attribute human characteristics to non-human entities.
 It has its (origins in) (basic misunderstandings） about how inanimate objects function.
 It demonstrates a key difference between human intelligence and machine intelligence.
 
38What potential advance does Rees see as a cause for concern? C
 robots (outnumbering) people
 robots having abilities which humans do not
 artificial intelligence developing independent thought
 artificial intelligence taking over every aspect of our lives
 
39What does Wolpert emphasise in his response to the question about science fiction?C B
 how science fiction influences our attitudes to robots
 how fundamental robots are to the (science fiction genre)
 how the image of robots in science fiction has changed over time
 how reactions to similar (portrayal)s of robots in science fiction may vary
 
40What is Richardson doing in her comment about reality and fantasy?C
 warning people not to [confuse] one [with] the other
 outlining 概述了ways in which one has impacted on the other
 recommending a change of approach in how people view them
 explaining why scientists have a different perspective on them from other people
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
Georgia O'Keeffe
charcoal  / ˈtʃɑːkəʊl / n.木炭  skyscraper / ˈskaɪskreɪpə(r) / n.摩天大楼
 
For seven decades, Georgia O'Keeffe (1887-1986) was a major figure in American art. Remarkably, she remained (independent from) shifting art trends and her work stayed true to her own vision, which was based on finding the essential,
abstract forms in nature. With exceptionally keen powers of observation and great finesse with a paintbrush, she recorded subtle nuances of colour, shape, and light that enlivened her paintings and attracted a wide audience.
在长达七十年的时间里，乔治亚·欧姬芙（1887-1986）一直是美国艺术界的核心人物。值得称道的是，她始终独立于不断变化的艺术潮流之外，其作品始终忠实于她自己的视觉理念，即寻找自然界中本质的、抽象的形式。凭借非凡敏锐的观察力和高超的画笔技艺，
她记录了色彩、形状和光线的微妙变化，这些元素赋予了她画作以生命力，并吸引了广泛的受众。

Born in 1887 near Sun Prairie, Wisconsin to cattle breeders Francis and Ida O'Keeffe, Georgia was (raised on) their farm along with her six siblings. #By the time she graduated from high school in 1905, she had determined to make her way as an artist.
She studied the techniques of traditional painting at the Art Institute of Chicago school (1905) and the Art Students League of New York (1907-8). After attending university and then training college, she became an art teacher and taught in elementary schools,
high schools, and colleges in Virginia, Texas, and South Carolina from 1911 to 1918.
1887年，欧姬芙出生于威斯康星州太阳草原市附近，父母是牛类繁育者弗朗西斯·欧姬芙和艾达·欧姬芙。她和六个兄弟姐妹一起在父母的农场里长大。到1905年高中毕业时，她就已经立志要成为一名艺术家。她曾在芝加哥艺术学院（1905年）和纽约艺术学生联盟（1907-1908年）学习传统绘画技巧。
在读完大学和培训学院后，她成为了一名美术老师，并于1911年至1918年期间在弗吉尼亚州、德克萨斯州和南卡罗来纳州的小学、高中和大学任教。

During this period, O'Keeffe began to experiment with creating abstract compositions in charcoal, and produced a series of innovative drawings that led her art in a new direction. She sent some of these drawings to a friend in New York, who showed them to art
collector and photographer Alfred Stieglitz in January 1916. Stieglitz was impressed, and exhibited the drawings later that year at his gallery on Fifth Avenue, New York City, where the works of many avant-garde artists and photographers were introduced to the American public.
在此期间，欧姬芙开始尝试用木炭创作抽象构图，并创作了一系列富有创新性的素描，这些作品将她的艺术推向了一个新的方向。她把其中的一些素描寄给了纽约的一位朋友，这位朋友在1916年1月将作品展示给艺术收藏家兼摄影师阿尔弗雷德·斯蒂格里茨。
斯蒂格里茨深受触动，并于同年晚些时候在纽约市第五大道的画廊展出了这些素描。在这里，许多前卫艺术家和摄影师的作品被首次介绍给美国公众。

With Stieglitz's encouragement and promise of financial support, O'Keeffe arrived in New York in June 1918 to begin a career as an artist. For the next three decades, Stieglitz (vigorously) promoted her work in twenty-two solo exhibitions and numerous group 
installations. The two were married in 1924. The ups and downs of their personal and professional relationship were recorded in Stieglitz's celebrated black-and-white portraits of O'Keeffe, taken (over the course of) twenty years (1917-37).
在斯蒂格里茨的鼓励和经济资助的承诺下，欧姬芙于1918年6月来到纽约，正式开启她的艺术家生涯。在接下来的三十年里，斯蒂格里茨在22场个展和无数次联展中大力推广她的作品。两人于1924年结婚。
他们个人关系与职业关系的起起落落，都被记录在斯蒂格里茨拍摄的长达二十年（1917-1937年）的著名黑白欧姬芙肖像摄影作品中。

By the mid-1920s, O'Keeffe was recognized as one of America's most important and successful artists, widely known for the (architectural) pictures that dramatically (depict) the soaring skyscrapers of New York. But most often, she painted botanical subjects, 
inspired by annual trips to the Stieglitz family summer home. In her (magnified) images depicting flowers, begun in 1924, O'Keeffe brings the viewer right into the picture.
到1920年代中期，欧姬芙已被公认为美国最重要且最成功的艺术家之一，因其戏剧性描绘纽约高耸摩天大楼的建筑画作而闻名遐迩。但最频繁的是，她会画植物题材，灵感来自每年去斯蒂格里茨家族夏日度假屋的旅行。在不晚于1924年开始创作的、描绘花卉的放大图像中，欧姬芙将观者直接带入了画面之中

(Enlarging the tiniest details to fill) an entire metre-wide (canvas) emphasized their shapes and lines and made them appear abstract. Such (daring compositions) helped establish O'Keeffe's reputation as an innovative modernist.
将最微小的细节放大以填满一整块一米宽的画布，强调了它们的形状和线条，并使它们显得抽象。这种大胆的构图帮助确立了欧姬芙作为创新现代主义者的声誉。

In 1929, O'Keeffe made her first extended trip to the state of New Mexico. It was a visit that had a (lasting impact) on her life, and an (immediate effect) on her work. Over the next two decades she made almost annual trips to New Mexico, staying up to six months 
there, painting in (relative solitude), then returning to New York each winter to exhibit the new work at Stieglitz's gallery. This pattern continued until she moved permanently to New Mexico in 1949.
1929年，欧姬芙第一次前往新墨西哥州进行了长时间的旅行。这次访问对她的生活产生了深远的影响，并对她的作品产生了立竿见影的效果。在接下来的二十年里，她几乎每年都会去新墨西哥州旅行，在那里生活长达六个月，
在相对孤独的环境中作画，然后每到冬天返回纽约，在斯蒂格里茨的画廊展出新作品。这种模式一直持续到1949年她永久搬到新墨西哥州。

There, O'Keeffe found new inspiration: at first, it was the numerous sun-bleached bones she came across in the state's (rugged terrain) that sparked her (imagination). Two of her earliest and most celebrated Southwestern paintings (exquisitely) reproduce a cow (skull)'s 
weathered surfaces, (jagged) edges, and irregular openings. Later, she also explored another variation on this theme in her large series of (Pelvis) pictures, which focused on the contrasts between convex and concave surfaces, and solid and open spaces.
在那里，欧姬芙找到了新的灵感：起初，是她在该州崎岖地形中偶遇的众多被阳光晒得发白的骨头，激发了她的想象力。她最早且最著名的两幅西南部画作精美地再现了牛头骨饱经风霜的表面、参差不齐的边缘以及不规则的孔洞。
后来，她还在她的大型《骨盆》（Pelvis）系列画作中探索了这一主题的另一种变体，该系列专注于凸面与凹面、实体空间与开放空间之间的对比。

However, it was the region's spectacular landscape, with its unusual (geological formation)s, vivid colours, (clarity of light), and (exotic vegetation), that held the artist's imagination for more than four decades. Often, she painted the (rock)s, cliffs, and mountains 
in (striking) close-up, just as she had done with her botanical subjects.
然而，是该地区壮丽的风景——其奇特的地质结构、鲜艳的色彩、清澈的光线和异国情调的植被——牢牢抓住了这位艺术家四十多年的想象力。通常，她会以惊人的特写来描绘岩石、悬崖和山脉，就像她对待植物题材时所做的那样。

O'Keeffe eventually owned two homes in New Mexico - the first, her summer (retreat) at Ghost Ranch, was (nestled) beneath 200-metre cliffs, while the second, used as her winter residence, was in the small town of Abiquiú. While both locales provided (a wealth of)
imagery for her paintings, one feature of the Abiquiú house - the large walled patio with its black door - was particularly inspirational. In more than thirty pictures between 1946 and 1960, she (reinvent)ed the patio into an abstract arrangement of geometric shapes.
欧姬芙最终在新墨西哥州拥有了两处住宅——第一处是她在幽灵牧场（Ghost Ranch）的夏日静修处，坐落在200米高的悬崖下；而第二处位于阿比休（Abiquiú）小镇，用作她的冬日居所。虽然这两个地方都为她的画作提供了丰富的意象，
但阿比休住宅的一个特征——带有黑门的大型围墙天井——尤其具有启发性。在1946年至1960年间的三十多幅画作中，她将天井重新塑造为几何形状的抽象排列。

From the 1950s into the 1970s, O'Keeffe travelled widely, making trips to Asia, the Middle East, and Europe. Flying in planes inspired her last two major series - (aerial views) of rivers and expansive paintings of the sky viewed from just above clouds. 
In both series, O'Keeffe increased the size of her canvases, sometimes to mural proportions, reflecting perhaps her newly expanded view of the world. When in 1965 she successfully translated one of her cloud (motif)s to a (monumental canvas) measuring 6 metres in 
length (with the help of assistants), it was an enormous challenge and a special (feat) for an artist nearing eighty years of age.
从1950年代到1970年代，欧姬芙广泛游历，前往亚洲、中东和欧洲旅行。乘飞机飞行激发了她最后两个主要系列的灵感——河流的鸟瞰图以及从云层上方俯瞰的天空的壮阔画作。在这两个系列中，欧姬芙都增大了画布的尺寸，有时甚至达到了壁画的规模，这也许反映了她新开拓的世界观。
1965年，当她在助手的帮助下，成功地将其中一个云朵图案转化到一块长达6米的巨型画布上时，这对于一位年近八十的艺术家来说是一个巨大的挑战，也是一项特殊的壮举。

The last two decades of the artist's life were relatively unproductive as ill health and blindness hindered her ability to work. O'Keeffe died in 1986 at the age of ninety-eight, but her rich (legacy) of some 900 paintings has continued to attract subsequent 
generations of artists and art lovers who derive inspiration from these very American images.
这位艺术家生命的最后二十年相对没有什么产出，因为身体抱恙和失明阻碍了她的工作能力。欧姬芙于1986年去世，享年98岁，但她留下的约900幅画作的丰富遗产继续吸引着后代的艺术家和艺术爱好者，他们从这些极具美国特色的图像中汲取灵感。
Choose ONE WORD ONLY from the passage for each answer.

The life and work of Georgia O'Keeffe
• studied art, then worked as a 1 teacher in various places in the USA
• created drawings using 2charcoal which were exhibited in New York City
• moved to New York and became famous for her paintings of the city's 3skyscrapers
• produced a series of innovative close-up paintings of 4 flowers
• went to New Mexico and was initially inspired to paint the many 5bones that could be found there
• continued to paint various features that together formed the dramatic 6 landscape of New Mexico for over forty years
• travelled widely by plane in later years, and painted pictures of clouds and 7rivers seen  

8Georgia O'Keeffe's style was greatly influenced by the changing fashions in art over the seven decades of her career.C B
 TRUE
 FALSE
 NOT GIVEN
9When O'Keeffe finished high school, she had already (made her mind up about) the career that she wanted.A
 TRUE
 FALSE
 NOT GIVEN
10Alfred Stieglitz first discovered O'Keeffe's work when she sent some abstract drawings to his gallery in New York City.A B
 TRUE
 FALSE
 NOT GIVEN
11O'Keeffe was (the subject of) Stieglitz's photographic work for many years.A
 TRUE
 FALSE
 NOT GIVEN
12O'Keeffe's paintings of the patio of her house in Abiquiú were among the artist's favourite works.A C
 TRUE
 FALSE
 NOT GIVEN
13O'Keeffe produced (a greater quantity of) work during the 1950s to 1970s than at any other time in her life.C
 TRUE
 FALSE
 NOT GIVEN
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
(Adapting to适应) the effects of climate change

A    All around the world, nations are already preparing for, and adapting to, climate change and its impacts. Even if we stopped all CO2 emissions tomorrow, we would continue to see the impact of the CO2 already released since industrial times,
with scientists forecasting that global warming would continue for around 40 years. In the meantime, ice caps would continue to melt and sea levels rise. Some countries and regions will suffer more extreme impacts from these changes than others. 
It's in these places that innovation is thriving.
在全世界范围内，各国已经开始为气候变化及其影响做准备并逐步适应。即使我们明天停止所有二氧化碳（$CO_2$）的排放，自工业时代以来已经释放的二氧化碳所带来的影响仍将持续，科学家预测全球变暖还将持续大约40年。
在此期间，冰帽将继续融化，海平面也将继续上升。某些国家和地区遭受这些变化带来的极端影响会比其他地方更为严重。正是这些地方，创新正在蓬勃发展。

B    In Miami Beach, Florida, USA, seawater isn't just breaching the island city's walls, it's seeping up through the ground, so the only way to save the city is to lift it up above sea level. Starting in the lowest and most vulnerable neighbourhoods, 
(roads have been raised by as much as 61 centimetres). The elevation work was carried out as part of Miami Beach's ambitious but much-needed stormwater-management programme. In addition to the (road adaptations), the city has (set up) new pumps that can remove
up to 75,000 litres of water per minute. In the face of floods, climate-mitigation strategies have often been overlooked, says Yanira Pineda, a senior sustainability coordinator. She knows that they're essential and that the job is (far from over). 'We know 
that in 20, 30, 40 years, we'll need to go back in there and adjust to the changing environment,' she says.
在美国佛罗里达州迈阿密海滩，海水不仅突破了这座岛屿城市的防波堤，还在从地下渗透出来，因此拯救这座城市的唯一方法就是将其抬高到海平面以上。从地势最低、最脆弱的社区开始，道路已经被抬高了多达61厘米。这项抬高工程是作为迈阿密海滩雄心勃勃但又迫切需要的雨水管理计划的一部分而开展的。
除了道路改造之外，该市还安装了新型水泵，每分钟可抽排多达75,000升的水。高级可持续发展协调员亚妮拉·皮内达（Yanira Pineda）表示，面对洪水，气候缓解策略常常被忽视。她深知这些策略至关重要，而且这项工作远未结束。她说：“我们知道，在20年、30年、40年后，我们还需要回到那里，去适应不断变化的环境。”

C    Seawalls are a staple strategy for many coastal communities, but on the soft, muddy northern shores of Java, Indonesia, they frequently collapse, further exacerbating coastal erosion. There have been many attempts to restore the island's coastal mangroves: 
ecosystems of trees and shrubs that help defend coastal areas by trapping sediment in their net-like root systems, elevating the sea bed and dampening the energy of waves and tidal currents. But Susanna Tol of the not-for-profit organisation Wetlands International 
says that, while hugely popular, (the majority of) mangrove-planting projects fail. So, Wetlands International (started out) with a different approach, building semi-permeable dams, made from bamboo poles and brushwood, to mimic the role of mangrove roots and create
(favourable conditions) for mangroves to (grow back) naturally. The programme has seen (moderate success), mainly in areas with less subsidence. 'Unfortunately, traditional infrastructure is often single-solution focused,' says Tol. 'For long-term success, it's critical
that we (transition towards multifunctional approaches) that embed natural processes and that engage and benefit communities and local decision-makers.'
防波堤是许多沿海社区的主要应对策略，但在印度尼西亚爪哇岛柔软泥泞的北部海岸，防波堤经常坍塌，进一步加剧了海岸侵蚀。人们曾多次尝试恢复该岛的沿海红树林：红树林是由树木和灌木组成的生态系统，它们通过网状的根系拦截沉积物、抬高海床并减弱海浪和潮汐流的能量，从而帮助保护沿海地区。
但非营利组织“国际湿地”（Wetlands International）的苏珊娜·托尔（Susanna Tol）表示，尽管红树林种植项目大受欢迎，但其中绝大多数都失败了。因此，国际湿地组织另辟蹊径，用竹竿和碎木搭建了半渗透堤坝，以模拟红树林根系的作用，为红树林的自然再生创造有利条件。该计划取得了初步成功，
主要是在地面沉降较轻的地区。托尔说：“不幸的是，传统的基础设施往往只专注于单一的解决方案。为了获得长期的成功，至关重要的一步是向多功能方法转变，这些方法需要融入自然过程，并让社区和地方决策者参与进来并从中受益。”

D    As the floodwaters rose in the rice fields of the Mekong Delta in September 2018, four small houses rose with them. Homes in this part of Vietnam are traditionally built on stilts but these ones had been built to float. The modifications were made by the 
Buoyant Foundation Project, a not-for-profit organisation that has been researching and (retrofitting) amphibious(/æmˈfɪbiəs/) houses since 2006. 'When I started this,' explains founder Elizabeth English, 'climate change was not (on the tip of everybody's tongue), but this 
technology is becoming necessary in places that didn't previously need it.' It's much cheaper than permanently elevating houses, English explains - about (a third of) what it would cost to completely replace a building's foundations. It also avoids the problem 
of taller houses being at greater risk from wind damage. Another plus (comes from the fact源于这样一个事实 that) amphibious structures can be sensitively adapted to meet cultural needs and match the kind of houses that are already common in a community.
2018年9月，随着湄公河三角洲稻田中的洪水上涨，四座小屋也随之升起。越南这一地区的房屋传统上是建在桩柱（吊脚楼）上的，但这几座房屋被改造成了可以漂浮的样式。这些改造是由非营利组织“漂浮基金会项目”（Buoyant Foundation Project）完成的，该组织自2006年以来一直致力于研究和改造两栖房屋。
创始人伊丽莎白·英格利希（Elizabeth English）解释说：“当我开始做这个项目时，气候变化还没有成为人尽皆知的话题，但现在这项技术在以前不需要它的地方正变得必不可少。”英格利希解释道，这比永久抬高房屋要便宜得多——大约只需彻底更换建筑物地基费用的三分之一。它还避免了房屋过高从而面临更大风灾风险的问题。
另一个优点是，两栖结构可以灵活调整以满足文化需求，并与社区中已经普遍存在的房屋类型相匹配。

E    Bangladesh is especially vulnerable to climate change. Most of the country is (less than) a metre above sea level and 80 per cent of its land lies on floodplains. 'Almost 35 million people living on the (coastal belt) of Bangladesh are currently affected by 
soil and water salinity,' says Raisa Chowdhury of the international development organisation ICCO Cooperation. Rather than (fighting against) it, one project is helping communities adapt to salt-affected soils. ICCO Cooperation has been working with 10,000 
farmers in Bangladesh to start cultivating naturally salt-tolerant crops in the region. Certain varieties of carrot, potato, kohlrabi, cabbage and beetroot have been found to be better suited to salty soil than the rice and wheat that is typically grown 
there. Chowdhury says that the results are very visible, comparing (a barren plot of land) to the 'beautiful, lush green vegetable garden' sitting beside it, in which he and his team have been working with the farmers. Since the project began, farmers trained
in saline agriculture have reported increases of two to three more harvests per year.
孟加拉国特别容易受到气候变化的影响。该国大部分国土海拔不足一米，80%的土地位于洪泛区。国际发展组织“ICCO合作组织”的赖萨·乔杜里（Raisa Chowdhury）表示：“目前，生活在孟加拉国沿海地带的近3500万人正受到土壤和水质盐碱化的影响。”一个项目并没有盲目对抗这一现状，而是正在帮助社区适应受盐碱化影响的土壤。
ICCO合作组织一直与孟加拉国的10,000名农民合作，开始在该地区种植天然耐盐作物。人们发现，某些品种的胡萝卜、土豆、苤蓝（大头菜）、卷心菜和甜菜根比当地通常种植的水稻和小麦更适合盐碱化土壤。乔杜里说，其成果是清晰可见的，他将一片荒芜的土地与旁边他和团队与农民共同经营的“美丽、郁郁葱葱的绿色菜园”进行了对比。
自该项目启动以来，接受过盐碱农业培训的农民报告称，每年可多收获两到三次。

F    Greg Spotts from Los Angeles (LA) in the USA is chief sustainability officer of the city's street services department. He leads the Cool Streets LA programme, a series of (pilot projects), which include the planting of trees and the installation of a 
'cool pavement' system, designed to help reach the city's goal of bringing down its average temperature by 1.5℃. 'Urban cooling is (literally a matter of) (life and death) for our future in LA,' says Spotts. Using a Geographic Information System data mapping 
tool, the programme identified streets with low tree canopy cover in three of the city's neighbourhoods and covered them with a light-grey, light-reflecting (coating), which had already been shown to lower road surface temperature in Los Angeles by 6℃. 
Spotts says one of these streets, in the Winnetka neighbourhood of San Fernando Valley, can now be seen as a (pale crescent), the only cool spot on an otherwise red thermal image, from the International Space Station.
来自美国洛杉矶（LA）的格雷格·斯波茨（Greg Spotts）是该市街道服务部门的高级可持续发展官。他领导了“凉爽街道洛杉矶”（Cool Streets LA）计划，这是一系列的试点项目，其中包括种植树木和安装“凉爽路面”系统，旨在帮助实现该市将平均温度降低 1.5℃ 的目标。
斯波茨说：“对于我们在洛杉矶的未来而言，城市降温确实是一件生死攸关的大事。”该计划利用地理信息系统（GIS）数据地图绘制工具，识别出了该市三个社区中树冠覆盖率较低的街道，并在其表面涂上了一层浅灰色、具有反光功能的涂层，该涂层此前已被证实能让洛杉矶的路面温度降低 6℃。斯波茨表示，
在圣费尔南多谷温内特卡社区的一条此类街道，现在从国际空间站拍摄的热成像图上看去，呈现为一个苍白的新月形，成为在一片红色热像中唯一的凉爽区域。
Reading Passage 2 has six paragraphs, A-F.
Which paragraph contains the following information?
1 how a type of plant (functions as) a natural protection for coastlines C
2 a prediction about how long it could take to stop noticing the effects of climate changeB A
3 a reference to the fact that a solution is particularly cost-effective划算的F D
4 a mention of a technology used to locate areas most (in need of) intervention D F

18 The stormwater-management programme in Miami Beach has involved the installation of efficient pumps18.
19 The construction of 19(dam)s was the first stage of a project to ensure the success of mangroves in Indonesia.
20 As a response to rising floodwaters in the Mekong Delta, a not-for-profit organisation has been building houses that can 20float.
21 Rising sea levels in Bangladesh have made it necessary to introduce various 21(crop)s that are suitable for areas of high salt content.
22 A project in LA has increased the number of 22trees on the city's streets. 

Look at the following statements (Questions 23-26) and the list of people below. 
Match each statement with the correct person, A-E.
23. It is essential to adopt strategies which involve and help residents of the region.D B
24. Interventions which reduce heat are absolutely vital for our survival in this location.E
25. More work will need to be done in future decades to deal with the impact of rising water levels.   A
26. The number of locations requiring action to adapt to flooding has grown in recent years.   C

A Yanira Pineda
B Susanna Tool
C Elizabeth English
D Raisa Chowdhury
E Greg Spotts

###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
A new role for livestock guard dogs

Livestock guard dogs, traditionally used to protect farm animals from predators, are now being used to protect the predators themselves

A    For thousands of years, livestock guard dogs worked alongside shepherds to protect their sheep, goats and cattle from predators such as wolves and bears. But in the 19th and 20th centuries, when such predators were largely exterminated, 
most guard dogs lost their jobs. In recent years, however, as increased efforts have been made to protect wild animals, predators have become more widespread again. As a result, farmers once more need to protect their livestock, and guard dogs are enjoying an 
unexpected revival. 

B    Today there are around (50 breeds of) guard dogs (on duty) in various parts of the world. These dogs are raised from an early age with the animals they will be watching and eventually these animals become the dog's family. The dogs will place 
themselves between the livestock and any threat, barking loudly. If necessary, they will (chase away) predators, but often their (mere presence) is sufficient. 'Their initial training is to make them understand that (livestock is going to be their life),
' says 😀Dan Macon, a shepherd with three guard dogs. 'A (fluffy white puppy) is fun to be around, but too much human affection makes it a great dog for guarding the front porch, rather than a great livestock guard dog.'

C    The evidence indicates that guard dogs are highly effective. For example, in Portugal, biologist 😀Silvia Ribeiro has found that more than 90 percent of the farmers participating in a programme to train and use guard dogs to protect their herds 
against attack from wolves (rate the performance of) the dogs as very good or excellent. In a study carried out in Australia by 😀Linda van Bommel and 😀Chris Johnson at the University of Tasmania, more than 65 percent of herders reported that (predation)
stopped completely after they got the dogs, and almost (all the rest) saw a decrease in attacks. 'If they are managed and used properly, livestock guard dogs are the most efficient control method that we have in terms of the amount of livestock that they 
save from predation,' says 😀van Bommel.

D    But today's guard dogs also have a new role - to help preserve the predators. It is hoped that reductions in livestock losses can make farmers more tolerant of predators and less likely to kill them. In Namibia, more than 90 percent of 😀cheetahs😀 live
outside protected areas, close to humans raising livestock. As a result, the cheetahs are often held responsible for animal losses, and large numbers have been killed by farmers. When guard dogs were introduced, more than 90 per cent of farmers reported a
dramatic reduction in livestock losses, and said that as a result they were less likely to kill predators. 😀Julie Young, at Utah State University in the US, believes this result applies widely. 'There is (common ground) from the livestock perspective and from 
the conservation perspective,' she says. 'If ranchers don't have a dead cow, they will not make a call to (apply for a permit) to kill a wolf.'

E    Looking at all the published evidence, 😀Bethany Smith at Nottingham Trent University in the UK found that up to 88 per cent of farmers said they no longer killed predators after using dogs - but warned that such self-reported results must be taken with
a pinch of salt. (What's more), it is possible that livestock guard dogs (merely displace predators) to unprotected neighbouring properties, where their fate isn't recorded. 'In some regions, we work with almost every farmer, but in others only one or two have dogs,'
says 😀Ribeiro. 'If we are not working with everybody, we are transferring the wolf pressure to the neighbour's herd and he can use (poison) and kill an entire pack of wolves.'

F    Another concern is whether there may be (unintended) ecological effects of using guard dogs. Studies suggest that reducing deaths of one type of predator may have a negative impact on other species. (The extent of this problem isn't known), but the consequences
are clear in Namibia. Cheetahs aren't the only species that cause sheep and goat losses there: other predators also attack livestock. In 2015, researchers reported that (in spite of) the impact farmers obtaining guard dogs had on cheetahs, the number of jackals killed
by dogs and people actually increased. Guard dogs have other ecological impacts too. They have been found to spread diseases to wild animals, including (endangered) Ethiopian wolves. They may also (compete with other carnivores) for food. And by creating a 'landscape of fear', 
their mere presence can influence the behaviour of prey animals.
另一个令人担忧的问题是，使用守护犬是否会产生意想不到的生态影响。研究表明，减少某一种掠食者的死亡可能会对其他物种产生负面影响。这一问题的严重程度尚不清楚，但在纳米比亚，其后果是显而易见的。在当地，猎豹并非唯一导致绵羊和山羊损失的物种：其他掠食者也会袭击牲畜。2015年，研究人员报告称，
尽管农民获得守护犬对猎豹产生了积极影响，但被狗和人类杀死的胡狼数量实际上增加了。守护犬还有其他的生态影响。人们发现它们会向野生动物传播疾病，包括濒危的埃塞俄比亚狼。它们还可能与其他肉食动物竞争食物。并且，通过创造一种“恐惧景观”，它们仅仅是存在就能影响猎物动物的行为。
G    The evidence so far, however, indicates that these consequences aren't always negative. Guard dogs can deliver unexpected benefits by protecting vulnerable wildlife from predators. For example, their presence has been found to protect birds which build their nests 
on the ground in fields, where foxes would normally (raid) them. Indeed, Australian researchers are now using dogs to enhance (biodiversity) and create (refuge)s for species threatened by predation. So if we can get this right, there may be a (bright future) for guard dogs in
promoting (harmonious coexistence) between humans and wildlife.
然而，迄今为止的证据表明，这些后果并不总是否面的。守护犬可以通过保护脆弱的野生动物免受掠食者侵害，带来意想不到的好处。例如，人们发现它们的存在可以保护那些在农田地面上筑巢的鸟类，否则狐狸通常会袭击这些鸟巢。事实上，澳大利亚的研究人员现在正利用狗来增强生物多样性，
并为受到捕食威胁的物种创造避难所。因此，如果我们能正确处理好这一点，守护犬在促进人类与野生动物和谐共处方面可能会有一个光明的未来。
Reading Passage 3 has seven paragraphs, A-G.

Which paragraph contains the following information?

NB   You may use any letter more than once.

27 an example of how one predator has been protected by the introduction of livestock guard dogs D
						
28 an optimistic suggestion about the possible positive developments in the use of livestock guard dogsG
						
29 a description of how the methods used by livestock guard dogs help to keep predators away B
						
30 claims by different academics that the use of livestock guard dogs is a successful way of protecting farmers’ herds C
						
31 a reference to how livestock guard dogs gain their skills  B
						
Questions 32-36

Look at the following statements (Questions 32-36) and the list of people below.

Match each statement with the correct person, A-E.


32. The use of guard dogs may save the lives of both livestock and wild animals.   D

33.Claims of a change in behaviour from those using livestock guard dogs may not be totally accurate.  B E

34. There may be negative results if the use of livestock guard dogs is not sufficiently widespread.  E B

35. Livestock guard dogs are the best way of protecting farm animals, as long as the dogs are appropriately handled.   C

36. Teaching a livestock guard dog how to do its work needs a different focus from teaching a house guard dog.  A
List of People
A   Dan Macon
B   Silvia Ribeiro
C   Linda van Bommel
D   Julie Young
E   Bethany Smith


In Namibia, livestock guard dogs have been used to protect domestic animals from attacks by cheetahs. This has led to a rise in the deaths of other predators, particularly  jackals 37. In addition, it has been suggested that the dogs could have  
38diseases which may affect other species, and that they may reduce the amount of  39 food  available to certain wild animals.
On the other hand, these dogs may help birds by protecting their nests. These might otherwise be threatened by predators such as  foxes40 .
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
How stress affect our judgement
压力如何影响我们的判断

「Q27」 Some of the most important decisions of our lives occur while we're feeling stressed and anxious. From medical decisions to financial and professional ones, we are all sometimes 
required to (weigh up) information under stressful conditions.
But do we become better or worse at processing and using information under such circumstances?
我们生命中的一些最重要的决定往往是在感到压力和焦虑时作出的。在从就医到财务和职场的种种决策中，我们每个人通常都需要在承受压力的情况下权衡信息。然而在这样的场景下，我们处理和利用信息的能力是变好还是变差了呢？

My colleague and I, both neuroscientists, wanted to investigate how the mind operates under stress, so we visited some local fire stations. Firefighters' workdays (vary quite a bit). Some are 
pretty relaxed; they'll spend their time washing the (truck), cleaning equipment, cooking meals and reading. Other days can be (hectic), with numerous life-threatening (incidents) to attend to(处理); 
they'll enter burning homes to rescue trapped residents, and assist with medical emergencies. 「Q28」 These ups and downs presented the perfect setting(n 环境，场景) for an experiment on how people's 
ability to use information changes when they feel under pressure.
我和我的同事都是神经科学家，我俩打算研究大脑如何在压力下运转，于是走访了几个本地消防站。消防员的工作日可以存在相当大的不同。一些日子是非常放松的；他们的时间花在清洗卡车、清洁设备、做做饭和读读书上。
另一些日子则可能十分繁忙，有无数威胁生命的事件需要处理；他们会进入熊熊燃烧的住家去拯救困住的居民，并协助紧急就医。这些忙闲时刻的剧烈波动提供了做实验的理想场景，可以展现出人们在感受到压力时运用信息的能力如何发生变化。

「Q35」 We found that (perceived) threat acted as a trigger for a stress reaction that made the task of processing information easier for the firefighters - but only as long as it (conveyed) bad news.
我们发现，对威胁的感知触发了一种压力反应，使得加工处理信息的任务对于消防员来说更容易了——但这仅仅是在坏消息传来的时候。

「Q35」 This is how we arrived at these results. 「Q29」 We asked the firefighters to estimate their likelihood of experiencing 40 different (adverse) events in their life, such as being 
involved in an accident or becoming (a victim of card fraud). 
We then gave them either good news (that their likelihood of experiencing these events was lower than they'd thought) or bad news (that it was higher) and asked them to provide new estimates.
我们是这样得出结论的。先让消防员预估一下他们在生活中遭遇40件不同恶性事件的几率，例如被卷入一场事故或成为银行卡欺诈事件的受害者。接下来再告诉他们好消息（他们遭遇这些事件的几率要低于他们所想的那样）
或坏消息（几率更高），然后让他们重新进行估计。

「Q35」 「Q31」 People are normally quite optimistic - they will ignore bad news and embrace the good. This is what happened when the firefighters were relaxed; 
「Q32」 but when they were under stress, a different pattern emerged. Under these conditions, 
they became hyper-(vigilant) to bad news, even when it had nothing to do这无关紧要 with their job (such as learning得知 that the likelihood 可能性 of card fraud was higher than they'd thought), 
and altered their beliefs in response. 
「Q33」 In contrast, stress didn't change how they responded to good news (such as learning that the likelihood of card fraud was lower than they'd thought).
人们通常都是相当乐观的——他们会忽略坏消息而拥抱好消息。当消防员处于放松状态时就是这样；但当他们置身压力下，一种不同的反应模式就出现了。在这些压力状态下，他们会对坏消息高度警觉，
即使这与他们的工作毫无关系（例如得知银行卡欺诈案件的发生几率比他们以为的要高），
而且会相应改变自己的想法。相较之下，压力并不会改变他们对好消息（例如得知银行卡欺诈的几率比自己以为的低）作出的反应。

「Q35」 Back in our lab, we observed the same pattern 「Q34」 in students who were told they had to give a surprise public speech, which would be judged by a panel, 
recorded and posted online. Sure enough 果然，不出所料 , their cortisol levels spiked, their heart rates
went up and they suddenly became better at processing unrelated, yet alarming令人担忧的，令人恐惧的, information about rates of disease and violence.
回到实验室场景下，我们在一群学生身上看到了相同的模式，他们被告知要临时准备一场公开演讲，且还会被一个专家组进行评判、录像并发表在网上。理所当然地，他们的(皮质醇水平飙升)、心率急剧加速，
突然就变得更善于处理一些虽然不相关但令人惊慌的关于疾病和暴力几率的消息了。

「Q35」 「Q30」 When we experience stressful events, a physiological生理的 change is triggered that causes us to take in warnings and focus on what might go wrong. Brain imaging 
reveals that this 'switch' is related to a sudden boost in a neural signal important 
for learning, specifically in response to unexpected warning signs, such as faces expressing fear.
当我们体验着压力事件时，一项生理变化就被触发了，这使得我们充分接收警报并聚焦在可能出问题的方面。脑成像显示：这种"切换"关联的是一种神经信号的激增，其对于学习行为至关重要，特别是在面对突发性警示信号如看到人们面露惊恐之时。

Such neural engineering could have helped prehistoric humans to survive. When our ancestors found themselves surrounded by hungry animals, they would have benefited from 
an increased ability to learn about hazards. 
In a safe environment, however, it would have been wasteful to be on high alert constantly. So, a neural switch that automatically increases or decreases our ability to process warnings in response to changes in our 
environment could have been useful. 
In fact, people with clinical depression and anxiety seem unable to switch away from a state in which they absorb all the negative messages around them.

这样的神经机制当初能帮助史前人类生存。当我们的祖先发现自己被饥饿野兽环伺时，如果辨识危险的能力有所提升当然就会受益。然而，在一个安全的环境中，如果总是处于预警状态就消耗太大了。因此，
拥有一个自动提升或降低我们在应对环境变化时处理警示信号的能力，会是很有用的。事实上，
那些确诊了抑郁和焦虑的人似乎无法调节关闭这种能力，因而时刻都在吸收周遭的所有负面信息。

It is also important to realise that stress travels rapidly from one person to the next. If a co-worker is stressed, we are more likely to tense up and feel stressed ourselves.
We don't even need to be in the same room with someone for their emotions to influence our behaviour. 
「Q36」 Studies show that if we observe positive feeds on social media, such as images of a pink sunset, we are more likely to post uplifting 令人振奋的；使人开心的 messages ourselves.
If we observe negative posts, such as complaints about a long queue at the coffee shop, we will in turn create more negative posts.
意识到这样一点也很重要：压力能迅速从一个人传导到另一个人身上。如果一个同事正在承受重压，我们自己也更容易紧张起来、感受到压力。我们甚至都不需要与某个人共处一室，他们的情绪就有可能影响到我们的行为
研究表明，如果我们在社交媒体上看到了积极消息，例如一场粉色落日的画面，
我们自己也有可能会发表更令人振奋的消息。如果我们看到了消极信息，例如抱怨在一家咖啡馆排了很长的队，那么我们自己也将随之发表更负面的言论。

In some ways, many of us now live as if we are in danger, constantly ready to tackle demanding emails and text messages, and respond to news alerts and comments on social media. 
「Q37」 Repeatedly checking your phone, according to a survey conducted by the American Psychological Association, is related to stress. In other words, a pre-programmed physiological reaction, 
which evolution has equipped us with to help us avoid famished predators, is now being triggered by an online post. 
Social media posting, according to one study, raises your pulse, makes you sweat, and enlarges your pupils more than most daily activities.

在某些方面，我们中的很多人现在活得就好像as if身处危险之中，时刻做好准备要应对棘手的邮件和短信，还要回应社交媒体上的消息提醒和评论。根据美国心理协会所做的一项调查，频繁查看手机的行为就与压力有关。换句话说，
这项预设在我们身体里的生理反应，原本是进化赋予我们以协助躲避饥肠辘辘猎食者的装备
，如今正在被线上消息所持续触发。根据一项研究，社交媒体上的各种信息比大多数日常活动都更能加速你的脉搏、引发你的出汗和放大你的瞳孔。

The fact that stress increases the likelihood that we will focus more on alarming messages, together with the fact that it spreads extremely rapidly, can create collective fear 
that is not always justified. 
「Q38」 After a stressful public event, such as a natural disaster or major financial crash, there is often a wave of alarming information in traditional and social media, which individuals become very aware of. 
But that has the effect of exaggerating existing danger.
「Q39」 And so, a reliable pattern emerges - stress is triggered, spreading from one person to the next, which temporarily enhances the likelihood that people will take in negative reports, which increases stress further. 
As a result, trips are cancelled, even if the disaster took place across the globe;  stocks are sold, even when holding on is the best thing to do.
压力会增加我们更关注警示类信息的可能性，并且它的传播速度极快，这二者结合起来可能会制造出并不总是正当合理的集体性恐慌。在一场高压公共事件——例如一场自然灾害或大规模金融崩盘——之后，传统媒体和社交媒体
上往往都会出现一波令人担忧的消息，而人们也会变得更加关注这些消息。
但这种现象的效果往往是夸大了实际风险。于是，一个固定的模式出现了—压力被触发，从一个人传到下一个人，暂时性地增加了人们更关注消极负面报导的可能性，这又进一步提升了压力值。结果就是，
出行计划被取消，哪怕这场灾难其实发生在地球的那一头；股票被抛售，哪怕持仓才是最明智的做法。

「Q40」 The good news, however, is that positive emotions, such as hope, are contagious 有感染力的too, and are powerful in inducing people to act to find solutions. Being aware of the close
relationship between people's emotional state and how they process information can help us frame our messages more effectively and become conscientious agents of change.

不过，好消息是：积极情绪（例如希望）也是有感染力的，而且能强有力地促使人们行动起来找到解决方案。明白在人们的情绪状态和处理信息方式这二者之间存在紧密关联，就能帮助我们更有效率地框定自己接收
到的各种信息，成为积极应对变化的行为主体。
#############################################################################################
27In the first paragraph, the writer introduces the topic of the text by   C
 defining some commonly used terms.
 questioning a widely held assumption.
 mentioning a challenge faced by everyone.
 specifying a situation which makes us most anxious.
############################################################################################# 
28What point does the writer make about firefighters in the second paragraph? A
 The regular changes of stress levels in their working lives make them ideal study subjects.
 The strategies they use to handle stress are of particular interest to researchers.
 The stressful nature of their job is typical of many public service professions.
 Their personalities make them especially well-suited to working under stress
 #############################################################################################
29What is the writer doing in the fourth paragraph?  D
 explaining their findings
 justifying their approach
 setting out their objectives
 describing their methodology
 #############################################################################################
30In the seventh paragraph, the writer describes a mechanism in the brain which  C
 enables people to respond more quickly to stressful situations.
 results in increased ability to control our levels of anxiety.
 produces heightened sensitivity to indications of external threats.
 is activated when there is a need to communicate a sense of danger.
#############################################################################################
Questions 31-35

Complete each sentence with the correct ending, A-G, below.

At times when they were relaxed, the firefighters usually  31A  B

The researchers noted that when the firefighters were stressed, they  32 G

When the firefighters were told good news, they always  33B F

The students' cortisol levels and heart rates were affected when the researchers 34E

In both experiments, negative information was processed better when the subjects 35 D

A. made them feel optimistic.

B. took relatively little notice of bad news.

C. responded to negative and positive information in the same way.

D. were feeling under stress.

E. put them in a stressful situation.

F. behaved in a similar manner, regardless of the circumstances.

G. thought it more likely that they would experience something bad.
##########################################################################
36The tone of the content we post on social media tends to reflect the nature of the posts in our feeds. A
 YES
 NO
 NOT GIVEN
37Phones have a greater impact on our stress levels than other electronic media devices. C
 YES
 NO
 NOT GIVEN
38The more we read about a stressful public event on social media, the less able we are to take the information in. A B
 YES
 NO
 NOT GIVEN
39Stress created by social media posts can lead us to take unnecessary precautions.A
 YES
 NO
 NOT GIVEN
40Our tendency to be affected by other people's moods can be used in a positive way.C A
 YES
 NO
 NOT GIVEN
###################################################################################################################################################################################################################
###################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
Questions 14-18
Reading Passage 2 has seven sections, A-G.
Which section contains the following information?
NB You may use any letter more than once.

14 reference to the research problems that arise from there being only a few surviving large elms C

15 details of a difference of opinion about the value of reintroducing elms to Britain G

16 reference to how Dutch elm disease was brought into Britain  B

17 a description of the conditions that have enabled a location in Britain to escape Dutch elm disease E

18 reference to the stage at which young elms become vulnerable to Dutch elm disease C

Questions 19-23
Look at the following statements (Questions 19-23) and the list of people below.
Match each statement with the correct person, A, B, or C.
NB You may use any letter more than once.

List of People:

A Matt Elliot

B Karen Russell

C Peter Bourne

Statements:

19 If a tree gets infected with Dutch elm disease, the damage rapidly becomes visible ________B

20 It may be better to wait and see if the mature elms that have survived continue to flourish __A______

21 There must be an explanation for the survival of some mature elms ____B____

22 We need to be aware that insects carrying Dutch elm disease are not very far away ______#C_

23 You understand the effect Dutch elm disease has had when you see evidence of how prominent the tree once was _____A___

Questions 24-26
Complete the summary below.
Choose ONE WORD ONLY from the passage for each answer.

Uses of a popular tree
For hundreds of years, the only tree that was more popular in Britain than elm was [ 24oak ]. 
Starting in the Bronze Age, many tools were made from elm and people also used it to make weapons.
In the 18th century, it was grown to provide wood for boxes and [ 25 flooring]. Due to its strength, elm was often used for mining equipment and the Cutty Sark's [ 26keel ] was also constructed from elm.
###########################################################################################################################################################################################################################################################################
Return of the elm: reintroducing the beloved tree to Britain

榆树回归：将心爱之树引回英国

Mark Rowe investigates attempts to reintroduce elms to Britain

Mark Rowe审视了在英国重新引入种植榆树的尝试

A    Around 25 million elms, accounting for 90% of all elm trees in the UK, died during the 1960s and '70s of Dutch elm disease. In the aftermath, the elm, once so dominant in the British landscape, was largely forgotten. However, there's now hope the elm may be reintroduced to the countryside of 
central and southern England. Any reintroduction will start from a very low base. &&&&&「Q23」 'The impact of the disease is difficult to picture if you hadn't seen what was there before,' says Matt Elliot of the Woodland Trust. 'You look at old photographs from the 1960s and it's only then that you 
realise the impact [elms had] ... They were significant, large trees ... then they were gone.'
A 大约2千5百万棵榆树，占英国所有榆树总量的90%，死于一九六零和七零年代中的那场荷兰榆树病。疫病过去后，曾在英国风景中占据了至高无上地位的榆树几乎被遗忘了。不过，现在英格兰中部和南部乡间有望重新引入种植榆树。任何复种都将从一个很低的基数起步。"如果你不曾见过之前的盛况，就无法理解这场疾病的影响力，"林地信托基金会的Matt Elliot如是说。
"你要看看1960年代的老照片，只有在那时你才能意识到[榆树所拥有的]地位 ...... 它们曾是如此雄伟壮观的林木 ...... 然后它们就消失了。"

B   The disease is caused by a fungus that blocks the elms' vascular (water, nutrient and food transport) system, causing branches to wilt and die. A first epidemic, which occurred in the 1920s, gradually died down,&&&&&「Q16」 but in the '70s a second epidemic was triggered by shipments of elm from Canada. 
The wood came in the form of logs destined for boat building and its intact bark was perfect for the elm bark beetles that spread the deadly fungus. This time, the beetles carried a much more virulent strain that destroyed the vast majority of British elms.
B 这场疾病的致病源是一种真菌，会阻隔榆树的脉管（水、养分和食物的运输）系统，造成枝干枯萎死亡。第一波疫病发生在1920年代，逐渐平息了下去，但是到了70年代，第二波疫病又被从加拿大运输榆木的行为所触发。这些木材是准备用作造船的原木，其完好无损的树皮是散播这种致命真菌的榆树皮甲虫的理想栖身之所。
这一次，这些甲虫携带了一种更加致命得多的菌株，毁灭了英国的绝大多数榆树。

C    Today, elms still exist in the southern English countryside but mostly only in low hedgerows between fields. 'We have millions of small elms in hedgerows but they get targeted by the q as soon as they reach a certain size,' says Karen Russell, co-author of the report 
'Where we are with elm'. 「Q18」 Once the trunk of the elm reaches 10-15 centimetres or so in diameter, it becomes a perfect size for beetles to lay eggs and for the fungus to take hold. Yet mature specimens have been identified, in counties such as Cambridgeshire, that are hundreds of years old, 
and have mysteriously escaped the epidemic.

C 今天，榆树仍然存在于英格兰南部乡村，但大多数都只生长在田野之间的矮树篱中。"我们有成百上千万长在树篱中的小榆树，可一旦它们长到了一定尺寸，就会被甲虫精准盯上，""我们对榆树知多少"这篇通讯文章的联合作者Karen Russell如是说。一旦榆树树干直径达到了10-15厘米左右，它就达成了甲虫产卵和真菌繁殖的绝好尺寸。
然而在例如剑桥郡这样的郡县内也找到了一些成熟的榆树，它们已生长了几百年之久，奇迹般地躲过了这场疫病。

The key, Russell says, is to identify and study those trees that have survived and work out why they stood tall when millions of others succumbed. &&&&&「Q14」 Nevertheless, opportunities are limited as the number of these mature survivors is relatively small. 
&&&&&「Q21」 'What are the reasons for their survival?' asks Russell. 'Avoidance, tolerance, resistance? We don't know where the balance lies between the three. I don't see how it can be entirely down to luck.'
Russel说，关键在于找到和研究这些幸存下来的树木，搞清楚它们之所以在成百万其它榆树都倒下之时仍能挺立的原因。然而，研究机会是有限的，因为这些幸存下来的成熟榆树数量较少。"它们的幸存原因是什么？"Russell追问道。"是能避开、耐受还是抵抗？我们不知道这三者之间的平衡在哪里。我不认为这纯粹是因为运气。"

D    &&&&&「Q24」 For centuries, elm ran a close second to oak as the hardwood tree of choice in Britain and was in many instances the most prominent tree in the landscape. Not only was elm common in European forests, it became a key component of birch, ash and hazel woodlands. 
The use of elm is thought to go back to the Bronze Age, when it was widely used for tools. Elm was also the preferred material for shields and early swords.  &&&&&「Q25」 In the 18th century, it was planted more widely and its wood was used for items such as storage crates and flooring. 
 &&&&&「Q26」 It was also suitable for items that experienced high levels of impact and was used to build the keel of the 19th-century sailing ship Cutty Sark as well as mining equipment.
D 若干世纪以来，榆树在英国作为硬木树种选择的地位仅略次于橡树，在很多时候也是户外风景中最惹人注目的树种。榆树不但在欧洲森林中随处可见，而且还是与白桦、白蜡和榛树所共同构成的混合林中一个不可或缺的组成部分。对榆木的利用被认为可以追溯到青铜时代，当时它被用来制作各种工具。人们在制造盾牌和早期佩剑时也更青睐使用它
。在18世纪这个树种得到了更广泛种植，其木材被用来制作诸如储物箱和铺地板等各类物品。它也很适合用来制作需要承受高冲击力的物品，曾被用于建造19世纪帆船Cutty Sark上的龙骨以及采矿设备。

E   Given how ingrained elm is in British culture, it's unsurprising the tree has many advocates. Amongst them is Peter Bourne of the National Elm Collection in Brighton. 'I saw Dutch elm disease unfold as a small boy,' he says. 'The elm seemed to be part of rural England,
but I remember watching trees just lose their leaves and that really stayed with me.'  &&&&&「Q17」 Today, the city of Brighton's elms total about 17,000. Local factors appear to have contributed to their survival. Strong winds from the sea make it difficult for the determined elm 
bark beetle to attack this coastal city's elm population. However, the situation is precarious.  &&&&&「Q22」 'The beetles can just march in if we're not careful, as the threat is right on our doorstep,' says Bourne.
E 理解了榆树在英国文化中的无处不在，也就不会惊讶于这种树有如此众多的拥趸。其中之一是位于布莱顿(Brighton)的国家榆树收藏协会成员Peter Bourne。"我在孩童时代目睹了荷兰榆树病的逐渐蔓延，"他说。"榆树就像英格兰乡村的一部分，但我记得眼看着这些树木失去了它们的叶子，那景象令我记忆犹新。"今天，
布莱顿全城的榆树总数达到了17,000棵。本地因素似乎助力了它们的生存。来自海上的强风使得那些顽固的榆树皮甲虫很难攻击这座海滨城市的榆树林。但是，局势仍不容乐观。"如果我们不小心应对，这些甲虫就会长驱直入，威胁近在咫尺，"Bourne如是说。

F   Any prospect of the elm returning relies heavily on trees being either resistant to, or tolerant of, the disease. This means a widespread reintroduction would involve existing or new hybrid strains derived from resistant, generally non-native elm species. 
A new generation of seedlings have been bred and tested to see if they can withstand the fungus by cutting a small slit on the bark and injecting a tiny amount of the pathogen. &&&&&「Q19」 'The effects are very quick,' says Russell. 'You return in four to six weeks and 
trees that are resistant show no symptoms, whereas those that are susceptible show leaf loss and may even have died completely.'
F 任何榆树回归的前景都在很大程度上有赖于树种要么可以抵抗、要么则可以承受荷兰榆树病。这就意味着，广泛的重新种植将会使用到已有或新培育的杂交树种，源自抗病的、基本并非本土原产的榆树品种。已经培育了新一代的小树苗并测试了它们能否扛住病菌，采用的方法是在树皮上切一个小口子、注射进少量的致病菌。
"效果立竿见影，"Russell说。"你在四到六周后再回来看，那些抗病的树木不会显示任何病症，而那些感染了病菌的树木则明显大量落叶且也许甚至已经完全死亡。"

G   All of this raises questions of social acceptance, acknowledges Russell. 'If we're putting elm back into the landscape, a small element of it is not native - are we bothered about that?' 「Q15」 For her, the environmental case for reintroducing elm is strong.
'They will host wildlife, which is a good thing.' Others are more wary. 'On the face of it, it seems like a good idea,' says Elliot. The problem, he suggests, is that, 'You're replacing a native species with a horticultural analogue*. You're effectively cloning.' 
There's also the risk of introducing new diseases. Rather than plant new elms, the Woodland Trust emphasises providing space to those elms that have survived independently. 「Q20」 'Sometimes the best thing you can do is just give nature time to recover ... over time, you might get resistance,' says Elliot.
G 所有这一切引发了社会认可的问题，Russell承认道。"如果我们要将榆树安放回乡村景致中去，它身上就会有一个元素不是土生土长的——我们会对此感到烦恼么？"对她来说，重新引入榆树的环保理由是强大的。"它们将会庇护一众野生生物，这是件好事。"另一些人则更加谨慎。"表面上看来，这似乎是个好主意，
"Elliot这样说。但他也指出，问题在于，"你是在用一种园艺模拟品替代一个本土品种。你实际上就是在克隆。"还存在引入新疾病的风险。比起种植新榆树，林地信托强调的是给那些凭一己之力存活下来的榆树提供空间。"有时候你所能做的最好的事，不过是给自然留出恢复的时间......一段时间之后，你也许就得到了抵抗力，"Elliot这样说。

* horticultural analogue: a cultivated plant species that is genetically similar to an existing species
###########################################################################################################################################################################################################################################################################
1There are other parrots that share the kakapo's inability to fly.
 TRUE
 FALSE *
 NOT GIVEN
2Adult kakapo produce chicks every year.
 TRUE
 FALSE *
 NOT GIVEN
3Adult male kakapo bring food back to nesting females.
 TRUE
 FALSE *
 NOT GIVEN
4The Polynesian rat was a greater threat to the kakapo than Polynesian settlers.
 TRUE
 FALSE 
 NOT GIVEN *
  5Kakapo were transferred from Rakiura Island to other locations because they were at risk from feral cats.
 TRUE  * 
 FALSE
 NOT GIVEN
6One Recovery Plan initiative that helped increase the kakapo population size was caring for struggling young birds.
 TRUE *
 FALSE
 NOT GIVEN

Questions 7-13

Complete the notes below.

Choose ONE WORD AND/OR A NUMBER from the passage for each answer.

New Zealand's kākāpō


 A type of parrot:

   • diet consists of fern fronds, various parts of a tree and 7bulbs#

 • nests are created in 8soil where eggs are laid.


 Arrival of Polynesian settlers

 • the 9feathers of the kākāpō were used to make clothes.


 Arrival of European colonisers
 10 deer were an animal which they introduced that ate the kākāpō's food sources.


 Protecting kākāpō

 • Richard Henry, a conservationist, tried to protect the kākāpō.

   • a definite sighting of female kākāpō on Rakiura Island was reported in the year 198011.

 • the Recovery Plan included an increase in funding12

 • a current goal of the Recovery Plan is to maintain the involvement of stakeholders 13in kākāpō protection.

The kākāpō is a nocturnal, flightless parrot that is critically endangered and one of New Zealand's unique treasures

The kākāpō, also known as the owl parrot, is a large, forest-dwelling bird, with a pale owl-like face. Up to 64 cm in length, it has predominantly yellow-green feathers, forward-facing eyes, a large grey beak, large blue feet, 
and relatively short wings and tail. ##It is the world's only flightless parrot, and is also possibly one of the world's longest-living birds, with a reported lifespan of up to 100 years.

Kākāpō are solitary birds and tend to occupy the same home range for many years. They forage on the ground and climb high into trees. They often leap from trees and flap their wings, but at best manage a controlled descent to the ground. 
They are entirely vegetarian, ##with their diet including the leaves, roots and bark of trees as well as bulbs, and fern fronds.

Kākāpō breed in summer and autumn, but only in years when food is plentiful. Males play no part in incubation or chick-rearing - females alone incubate eggs and feed the chicks. The 1-4 eggs are laid in ##soil, which is repeatedly 
turned over before and during incubation. The female kākāpō has to spend long periods away from the nest searching for food, which leaves the unattended eggs and chicks particularly vulnerable to predators.

Before humans arrived, kākāpō were common throughout New Zealand's forests. However, this all changed with the arrival of the first Polynesian settlers about 700 years ago. For the early settlers, the flightless kākāpō was easy prey. 
They ate its meat and used its ##feathers to make soft cloaks. With them came the Polynesian dog and rat, which also preyed on kākāpō. By the time European colonisers arrived in the early 1800s, kākāpō had become confined to the 
central North Island and forested parts of the South Island. The fall in kākāpō numbers was accelerated by European  colonisation. A great deal of habitat was lost through forest clearance, and introduced species such as
##deer depleted the remaining forests of food. Other predators such as cats, stoats and two more species of rat were also introduced. The kākāpō were in serious trouble.

In 1894, the New Zealand government launched its first attempt to save the kākāpō. Conservationist Richard Henry led an effort to relocate several hundred of the birds to predator-free Resolution Island in Fiordland. 
Unfortunately, the island didn't remain predator free - stoats arrived within six years, eventually destroying the kākāpō population. By the mid-1900s, the kākāpō was practically a lost species. Only a few clung to life 
in the most isolated parts of New Zealand.

From 1949 to 1973, the newly formed New Zealand Wildlife Service made over 60 expeditions to find kākāpō, focusing mainly on Fiordland. Six were caught, but there were no females amongst them and all but one died within 
a few months of captivity. In 1974, a new initiative was launched, and by 1977, 18 more kākāpō were found in Fiordland. However, there were still no females. In 1977, a large population of males was spotted in Rakiura - 
a large island free from stoats, ferrets and weasels. There were about 200 individuals, and in 1980 it was confirmed females were also present. These birds have been the foundation of all subsequent work in managing the species.

Unfortunately, predation by feral cats on Rakiura Island led to a rapid decline in kākāpō numbers. As a result, during 1980-97, the surviving population was evacuated to three island sanctuaries: Codfish Island, Maud 
Island and Little Barrier Island. However, breeding success was hard to achieve. Rats were found to be a major predator of kākāpō chicks and an insufficient number of chicks survived to offset adult mortality. By 1995, 
although at least 12 chicks had been produced on the islands, only three had survived. The kākāpō population had dropped to 51 birds. The critical situation prompted an urgent review of kākāpō management in New Zealand.

In 1996, a new Recovery Plan was launched, together with a specialist advisory group called the Kākāpō Scientific and Technical Advisory Committee and a higher amount of funding. Renewed steps were taken to control 
predators on the three islands. Cats were eradicated from Little Barrier Island in 1980, and possums were eradicated from Codfish Island by 1986. However, the population did not start to increase until rats were removed 
from all three islands, and the birds were more intensively managed. This involved moving the birds between islands, supplementary feeding of adults and rescuing and hand-raising any failing chicks.

After the first five years of the Recovery Plan, the population was on target. By 2000, five new females had been produced, and the total population had grown to 62 birds. For the first time, there was cautious optimism
for the future of kākāpō and by June 2020, a total of 210 birds was recorded.

Today, kākāpō management continues to be guided by the kākāpō Recovery Plan. Its key goals are: minimise the loss of genetic diversity in the kākāpō population, restore or maintain sufficient habitat to accommodate the 
expected increase in the kākāpō population, and ensure stakeholders continue to be fully engaged in the preservation of the species.

