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
