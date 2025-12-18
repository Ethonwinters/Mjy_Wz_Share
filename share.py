(meshgnnnn) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ conda list | grep pytorch
pytorch                   2.4.1           cpu_generic_py310h994918d_5    conda-forge
pytorch-cuda              12.1                 ha16c6d3_6    pytorch
pytorch-mutex             1.0                        cuda    pytorch
torchaudio                2.4.1               py310_cu121    pytorch
torchvision               0.19.1              py310_cu121    pytorch

(meshgnn) adminstrator@adminstrator-HP-Z8-G5-Workstation-Desktop-PC:~/GNN/meshGraphNets_pytorch$ conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia --override-channels
Channels:
 - pytorch
 - nvidia
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: failed

LibMambaUnsatisfiableError: Encountered problems while solving:
  - nothing provides libpng needed by torchvision-0.19.1-py310_cpu
  - nothing provides numpy needed by torchaudio-2.4.1-py310_cpu
  - nothing provides typing_extensions needed by pytorch-2.4.1-py3.10_cpu_0

Could not solve for environment specs
The following packages are incompatible
├─ pytorch ==2.4.1 * is not installable because it requires
│  └─ typing_extensions =* *, which does not exist (perhaps a missing channel);
├─ torchaudio ==2.4.1 * is not installable because it requires
│  └─ numpy =* *, which does not exist (perhaps a missing channel);
└─ torchvision ==0.19.1 * is not installable because it requires
   └─ libpng =* *, which does not exist (perhaps a missing channel).
