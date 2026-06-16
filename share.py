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
The table shows statistics about the amount of pupulation and compares the change of five districts population proportion in NewYork City over a period of 200 years.  
The tables show statistics about the total population and compare changes in the population proportions of five districts in New York City over a period of 200 years

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

海牛

Manatees, also known as sea cows, are aquatic mammals that belong to a group of animals called Sirenia. This group also contains dugongs. Dugongs and manatees look quite alike - they are similar in size, colour and shape, and both have flexible flippers for forelimbs. 
「Q1」 However, the manatee has a broad, rounded tail, whereas the dugong's is fluked, like that of a whale. There are three species of manatees: the West Indian manatee (Trichechus manatus), the African manatee (Trichechus senegalensis) and the Amazonian manatee (Trichechus inunguis).

海牛，也被称为海中牛，是隶属于海牛目的一类水生哺乳动物。这个类目中还包括儒艮。儒艮和海牛看起来很相似——他们在大小、颜色和体型上都差不多，也都有着灵活的鳍足作为上肢。不过，海牛有着宽阔圆润的尾巴，而儒艮的尾巴则是三角分叉的，就像鲸鱼那样。有三个品种的海牛：西印度海牛（Trichechus manatus）、非洲海牛（Trichechus senegalensis）
和亚马逊海牛（Trichechus inunguis）

Unlike most mammals, manatees have only six bones in their neck - most others, including humans and giraffes, have seven. This short neck allows a manatee to move its head up and down, but not side to side. 「Q2」 To see something on its left or its right, a manatee must turn its entire body, 
steering with its flippers. Manatees have pectoral flippers but no back limbs, only a tail for propulsion. They do have pelvic bones, however - a leftover from their evolution from a four-legged to a fully aquatic animal. Manatees share some visual similarities to elephants. Like elephants, manatees have thick, wrinkled skin. 
「Q3」 They also have some hairs covering their bodies which help them sense vibrations in the water around them.
与大多数哺乳动物不同，海牛的脖子只有六块骨头——而大多数其他动物，包括人类和长颈鹿，都有七块。这个短脖子使得海牛可以上下移动头部，但是无法从一侧平移向另一侧。要想看到自己左侧或右侧的任何景象，海牛需要转动整个身体，用鳍足来掌控方向。海牛有胸鳍但是没有后肢，只能用尾巴来产生推动力。不过，它们有着盆骨结构——这是从四脚动物到完全水生动物的进化遗存。
海牛与大象有着某些视觉上的相似度。就像大象那样，海牛也有着厚实起皱的皮肤。它们周身也长着一些毛发来帮助它们感知周围水域的波动。

「Q4」 Seagrasses and other marine plants make up most of a manatee's diet. Manatees spend about eight hours each day grazing and uprooting plants. They eat up to 15% of their weight in food each day. African manatees are omnivorous - studies have shown that molluscs and fish make up a small part of their diets. 
West Indian and Amazonian manatees are both herbivores.

海草及其它海洋植物构成了一头海牛食谱中的绝大部分。海牛一天中要花大约八个小时翻找啃食植物。它们每天的最大进食量可达身体重量的15%。非洲海牛是杂食性的——研究显示软体动物和鱼类组成了其食谱中的一小部分。西印度海牛和亚马逊海牛则都是素食动物。

Manatees' teeth are all molars - flat, rounded teeth for grinding food. Due to manatees' abrasive aquatic plant diet, these teeth get worn down and they eventually fall out, so they continually grow new teeth that get pushed forward to replace the ones they lose. 「Q5」 Instead of having incisors to grasp their food, 
manatees have lips which function like a pair of hands to help tear food away from the seafloor.
海牛的所有牙齿都是臼齿——用来研磨食物的扁平圆形齿。由于海牛要靠磨碎海洋植物进食，这些牙齿会不断磨损直至最终脱落，因此它们会不断向前长出新牙以替代失去的那些。海牛并没有门齿可用来固定住食物，它们嘴唇的功能就像一双手，用来将食物从海床上薅起拔走。

Manatees are fully aquatic, but as mammals, they need to come up to the surface to breathe. When awake, they typically surface every two to four minutes, but they can hold their breath for much longer. Adult manatees sleep underwater for 10-12 hours a day, but they come up for air every 15-20 minutes. Active manatees
need to breathe more frequently. 「Q6」 It's thought that manatees use their muscular diaphragm and breathing to adjust their buoyancy. They may use diaphragm contractions to compress and store gas in folds in their large intestine to help them float.
海牛完全生活在水中，但是作为哺乳动物，它们需要来到水面以上呼吸。在醒着时，它们通常会每隔两到四分钟浮出水面一次，但其实它们能憋气的时间比这长得多。成年海牛每天要在水下睡眠10-12小时，但它们每15-20分钟就上来透一次气。活泼的海牛需要更频繁地换气。通常认为海牛会运用其发达的横膈膜和呼吸来调节自身浮力。它们还可以收缩横膈膜，将空气压缩和储存在其大肠的褶皱中以帮助自己漂浮。

The West Indian manatee reaches about 3.5 metres long and weighs on average around 500 kilogrammes. 「Q7」 It moves between fresh water and salt water, taking advantage of coastal mangroves and coral reefs, rivers, lakes and inland lagoons. There are two subspecies of West Indian manatee: the Antillean manatee is found in
waters from the Bahamas to Brazil, whereas the Florida manatee is found in US waters, although some individuals have been recorded in the Bahamas. 「Q8」 In winter, the Florida manatee is typically restricted to Florida. When the ambient water temperature drops below 20℃, it takes refuge in naturally and artificially warmed
water, such as at the warm-water outfalls from powerplants.
西印度海牛体长可达3.5米，平均体重约为500公斤。它可在淡水和咸水环境间穿梭，充分享受海滨红树林和珊瑚礁、河流、湖泊及内陆泻湖等各种地利。西印度海牛有两个亚种：安替列（Antillean）海牛可见于从巴哈马到巴西的水域中，而佛罗里达海牛则分布在美国水域中，不过巴哈马也有过少量发现记录。在冬季，佛罗里达海牛的活动范围通常只局限于佛罗里达。当周遭环境温度掉到20度以下时
，它寻求庇护的地点就是各种天然或人工的温暖水域，例如发电厂附近暖意融融的排水口。

「Q9」 The African manatee is also about 3.5 metres long and found in the sea along the west coast of Africa, from Mauritania down to Angola. The species also makes use of rivers, with the mammals seen in landlocked countries such as Mali and Niger.
非洲海牛的体长也可达致约3.5米，遍布在从毛里塔尼亚一路南下到安哥拉的非洲西海岸海域。这个品种也可以生活在江河中，在例如马里和尼日尔这样四围都是陆地的国家中也能看到这种哺乳动物。

The Amazonian manatee is the smallest species, though it is still a big animal. It grows to about 2.5 metres long and 350 kilogrammes. Amazonian manatees favour calm, shallow waters that are above 23℃. This species is found in fresh water in the Amazon Basin in Brazil, as well as in Colombia, Ecuador and Peru.

亚马逊海牛是体型最小的品种，尽管它仍是一头大型动物。它能长到约2.5米，重350公斤。亚马逊海牛青睐水温在23度以上的平缓浅水区。在巴西的亚马逊盆地，以及哥伦比亚、厄瓜多尔和秘鲁的淡水中都能看到这个品种。

All three manatee species are endangered or at a heightened risk of extinction. The African manatee and Amazonian manatee are both listed as Vulnerable by the International Union for Conservation of Nature (IUCN). 「Q10」 It is estimated that 140,000 Amazonian manatees were killed between 1935 and 1954 for their meat,
fat and skin, with the latter used to make leather. In more recent years, African manatee decline has been tied to incidental capture in fishing nets and hunting. Manatee hunting is now illegal in every country the African species is found in.

所有这三个海牛品种都处在濒危之中或灭绝风险增高。非洲海牛和亚马逊海牛双双被世界自然保护联盟（IUCN）列为"脆弱"级别。据估计，在1935到1954年间有140,000只亚马逊海牛被捕杀是为了获取它们的肉、脂肪和皮，后者被用来制作皮革。更近些年来，非洲海牛数量的下降主要被归咎于渔网偶然捕获和故意捕猎。海牛捕猎行为现如今在每个能发现非洲海牛品种的国家中都是非法的。

「Q11」 The two subspecies of West Indian manatee are listed as Endangered by the IUCN. Both are also expected to undergo a decline of 20% over the next 40 years. 「Q12」 A review of almost 1,800 cases of entanglement in fishing nets and of plastic consumption among marine mammals in US waters from 2009 to 2020 found that 
at least 700 cases involved manatees. 「Q13」 The chief cause of death in Florida manatees is boat strikes. However, laws in certain parts of Florida now limit boat speeds during winter, allowing slow-moving manatees more time to respond.
西印度海牛的两个亚种均被世界自然保护联盟列为"濒危"级别。二者也都被预测在接下来的40年间数量将减少20%。对2009到2020年间美国水域中海洋哺乳动物被缠进渔网和误食塑料的将近1800起案例的回顾显示，至少有700起事故中波及到了海牛。佛罗里达海牛的主要死因是船只撞击。不过，佛罗里达部分地区的法律目前会限制船只在冬季的速度，为这些移动缓慢的海牛留出更多反应时间。


 Manatees
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
13There is some legislation in place which aims to reduce the likelihood of boat strikes on manatees in Florida. A
目前已经有一些法律法规出台，旨在降低佛罗里达州海牛遭受船只撞击（boat strikes）的可能性。
 TRUE
 FALSE
 NOT GIVEN























