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
###################################################################################################################################################################################################################
###################################################################################################################################################################################################################
27、解题思路：文章第一段通过描述人们在医疗、财务和职业等重要领域面临决策时感受到压力（stressed and anxious）的状态，指出我们所有人有时都必须在压力下权衡信息。这种普遍面临的困难处境对应了选项C中的“challenge”，作者以此引出全文关于压力如何影响判断的讨论。

对应文章：第1段整段。

避坑：不要误选B选项。虽然段末提出了一个问题，但作者并没有质疑某个“广泛持有的假设”，而是通过提问来引导读者思考。同时，选项D提到的“最焦虑的情况”在文中并未界定，文中只是列举了多种压力场景。
解析反馈
28、解题思路：题干询问第二段关于消防员（firefighters）的内容。文中描述了消防员工作状态的剧烈波动，从清闲的日常维护到高压的紧急救援。作者指出，这种“起伏（ups and downs）”为观察压力如何改变信息处理能力提供了“完美的实验环境（perfect setting）”。这直接对应了选项A，
即消防员工作压力的定期变化（regular changes）使其成为理想的研究对象（ideal study subject）。

对应文章：第2段：These ups and downs presented the perfect setting for an experiment on how people's ability to use information changes when they feel under pressure.

避坑：注意不要过度推断。虽然文中提到了消防员的工作内容，但并没有说他们的工作是公共服务行业的“典型代表”（选项C），也没有讨论他们的“性格特征”（选项D），重点在于其工作环境对科学实验的价值。
解析反馈
29、解题思路：题干问第4段写了什么。在第四段中，作者详细列举了与消防员（firefighters）进行实验的具体步骤：要求受试者估算概率、提供正面或负面信息反馈、再次要求估算。这种对实验过程、步骤和手段的详细记录，在学术写作中属于对“研究方法（methodology）”的描述，故选D。

避坑：区分“过程”与“结论”。很多考生会误选A（解释发现），但实际上第四段只讲了“怎么做”，而实验得出的“结论”（即压力下人们对坏消息更敏感）是在第五段才开始解释的。
解析反馈
30、解题思路：第七段解释了大脑中的一种生理机制。当压力发生时（stressful events），该机制会被触发（is triggered/switch），促使我们关注警告并留意可能出错的地方。文中的“neural signal boost”和“unexpected warning signs”对应了选项C中的“heightened sensitivity”和
“indications of external threats”，即增强了对外部威胁信号的敏感度。因此选C。

对应文章：第7段整段。

避坑：避开选项A的干扰。虽然该机制与压力有关，但文中强调的是对信息的“处理和关注（take in warnings）”，而非反应的“速度（more quickly）”。另外，选项D中的“communicate”意为交流传递，而文中讨论的是大脑内部的接收和处理机制。

31、解题思路：由firefighters和relaxed定位到文中第5段，可知对应的内容应是optimistic和ignore bad news、embrace the good，对应B选项“took relatively little notice of bad news。

对应文章：第5段：People are normally quite optimistic - they will ignore bad news and embrace the good. This is what happened when the firefighters were relaxed;

避坑：考生需要准确区分“放松”和“压力”两种对比状态下的不同反应。在放松状态下，人类天生的“乐观偏见”占据主导，因此会下意识地过滤掉负面预测，不要将其与压力状态下对坏消息的警觉反应混淆。
解析反馈
32、解题思路：紧接着上一题文中定位处即有under stress的情况，这时firefighters会became hyper-vigilant to bad news, even when...，对应G选项“认为更可能遭遇坏情况”。

对应文章：第5段：but when they were under stress, a different pattern emerged. Under these conditions, they became hyper-vigilant to bad news, even when it had nothing to do with their job (such as learning that the likelihood of card fraud was higher than they'd thought), and altered their beliefs in response.

避坑：理解“hyper-vigilant”是解题的关键。它不仅指注意力集中，更隐含了对风险评估的调整。压力作为一种触发机制，使大脑从乐观模式切换到了风险防范模式，从而提高了对负面事件发生概率的预期。
解析反馈
33、解题思路：紧接着上一题文中定位处即有good news，可知此时stress didn’t change，对应F选项。

对应文章：第5段：In contrast, stress didn't change how they responded to good news (such as learning that the likelihood of card fraud was lower than they'd thought).

避坑：本题的陷阱在于考生可能会主观推测压力会全方位改变人的认知。但文章强调了这种变化的“选择性”：压力只特异性地增强了对负面信息的吸收，而对正面信息的处理逻辑则保持原样，这种不对称性是实验的核心发现之一。
解析反馈
34、解题思路：第6段开始提到对学生的实验。可知当学生突然被告知要做公共讲话时(即处于stressful situation)，他们的cortisol levels spiked, their heart rates went up，对应选项E。

对应文章：第6段：in students who were told they had to give a surprise public speech, which would be judged by a panel, recorded and posted online. Sure enough, their cortisol levels spiked, their heart rates went up ...

避坑：解题时应聚焦于“被告知要演讲”这一瞬间触发的即时反应。虽然文章随后提到了他们处理信息能力的提升，但那是生理反应带来的结果。题干询问的是状态描述，因此最直接、最准确的对应点是文中描述的生理变化。
解析反馈
35、解题思路：答案解析：以firefighters和students为实验对象的两个实验在第3到第7段有陈述、分析、和总结，尤其在第6段首句指明是the same pattern，可知都是处于压力时对负面信息处理得更好，选D。
对应文章：第3、4、5、6、7段。

避坑：本题要求对两个实验对象进行横向总结。考生需要识别出文中“bad news”、“alarming information”与选项中“negative information”的同义替换关系。同时要理解文章的主旨：压力是一种进化而来的机制，旨在提高个体在危险环境中处理威胁信息效率。

36、解题思路：根据题干中的关键词 social media 和 feeds 定位到文章第九段最后两句。文中明确指出，如果我们看到社交媒体上的积极动态（如粉色日落的图片），就更有可能发布令人振奋的信息；反之，如果看到负面帖子（如抱怨排长队），则会相应地产生更多负面内容。这说明我们发布内容的基调反映了 
feeds 内容的性质，与题干描述完全一致，故判定为 YES。

对应文章：第9段：Studies show that if we observe positive feeds on social media, such as images of a pink sunset, we are more likely to post uplifting messages ourselves. If we observe negative posts, such as complaints about a long queue at the coffee shop, we will in 
turn create more negative posts.

避坑：在做此类题目时，要注意文中提到的“positive feeds”和“negative posts”与题干中“nature of the posts”的对应关系。不要被具体的例子（如日落或排队）干扰，而要抓住其背后的逻辑规律，即社交媒体上情绪和内容的传染性。只要文中逻辑方向与题干一致，即可判定为 YES。
解析反馈
37、解题思路：利用关键词 phones 定位到文章第十段第二句。该句提到根据美国心理学会的一项调查，频繁查看手机与压力有关。然而，纵观全文，作者仅讨论了手机使用与压力的相关性，并未将手机对压力水平的影响与其他电子媒体设备（如平板电脑、电脑或电视等）进行对比，也没有提到手机的影响是否“更大”。
由于文中缺失横向比较的信息，判定为 NOT GIVEN。

对应文章：第10段：Repeatedly checking your phone, according to a survey conducted by the American Psychological Association, is related to stress.

避坑：这是典型的“无中生有”型题目。考生容易根据生活常识认为手机确实比其他设备使用频率更高、带来的压力更大，从而先入为主地误选 YES。在雅思阅读中，必须严格遵循原文，文中缺失横向比较的信息，因此只能判定为 NOT GIVEN。
解析反馈
38、解题思路：根据题干 stressful public event 定位至第十一段。文中指出，在经历此类压力事件后，个人会变得“非常关注”（become very aware of）媒体上的警示信息，且压力会暂时“增加”（enhances）人们吸收负面报告的可能性。这与题干中“越读越无法吸收信息”（less able to take the information in）
的描述完全相反，故判断为 NO。

避坑：注意理解文中“take in”和“become aware of”的含义，它们都指向“获取/吸收信息”。题干设置了反向逻辑，将文中的“增强吸收能力”改成了“减弱吸收能力”。考生需要准确识别文中描述的压力反馈机制，即压力反而会让人对负面警示信息变得更加敏感而非迟钝。
解析反馈
39、解题思路：定位到第11段末尾。文中描述了一个恶性循环：压力在人与人之间传播，增强了人们吸收负面信息的可能性，进而导致人们做出过度反应。文章给出了具体例子：即使灾难发生在地球另一端也会取消行程，或者在应该持有时卖出股票。这些在客观上并非必须、甚至有损利益的行为，正对应了题干中的“不必
要的预防措施”（unnecessary precautions），信息一致，故答案为YES。

对应文章：第11段：After a stressful public event, ... ... And so, a reliable pattern emerges - stress is triggered, spreading from one person to the next, which temporarily enhances the likelihood that people will take in negative reports, which increases stress further. 
As a result, trips are cancelled, even if the disaster took place across the globe; stocks are sold, even when holding on is the best thing to do.

避坑：本题的难点在于对“unnecessary precautions”的同义替换理解。文中虽然没有直接出现这个词组，但通过“即使灾难在远方也取消行程”和“本该持有却卖出股票”这两个例子，暗示了这些行为是基于恐惧而非理性做出的过度防范。考生需要具备从具体案例中归纳抽象概念的能力。
解析反馈
40、解题思路：定位到文章最后一段。作者在结尾处给出了积极的观点，指出积极情绪（如希望）同样具有传染性，并且能强有力地诱导人们采取行动寻找解决方案。末段首句提到的 positive emotions 会引诱人们采取行动找到解决方法，正对应了题干中“他人的情绪可被积极运用”的描述。最后一句进一步提到，
意识到情绪状态与信息处理之间的紧密关系，可以帮助我们更有效地传达信息，成为积极的改变者。这证实了这种“受他人情绪影响”的心理倾向确实可以被正面、积极地利用，信息一致，故答案为YES。

对应文章：末段：The good news, however, is that positive emotions, such as hope, are contagious too, and are powerful in inducing people to act to find solutions.

避坑：考生可能会受前文大量关于压力、焦虑和负面影响的描述所误导，认为这种情绪传染倾向全是坏处。但雅思阅读文章往往在结尾处进行观点升华。看到“The good news”和“positive emotions”等词汇时，应敏锐察觉到作者论述方向的转变，从而准确匹配题干中的“positive way”。
###################################################################################################################################################################################################################
###################################################################################################################################################################################################################
How stress affects our judgement

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
they became hyper-(vigilant) to bad news, even when it had nothing to do with their job (such as learning得知 that the likelihood 可能性 of card fraud was higher than they'd thought), 
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
「Q37」 Repeatedly checking your phone, according to a survey conducted by the 
American Psychological Association, is related to stress. In other words, a pre-programmed physiological reaction, which evolution has equipped us with to help us avoid famished 
predators, is now being triggered by an online post. Social media posting, according
to one study, raises your pulse, makes you sweat, and enlarges your pupils more than most daily activities.

在某些方面，我们中的很多人现在活得就好像as if身处危险之中，时刻做好准备要应对棘手的邮件和短信，还要回应社交媒体上的消息提醒和评论。根据美国心理协会所做的一项调查，频繁查看手机的行为就与压力有关。换句话说，
这项预设在我们身体里的生理反应，原本是进化赋予我们以协助躲避饥肠辘辘猎食者的装备
，如今正在被线上消息所持续触发。根据一项研究，社交媒体上的各种信息比大多数日常活动都更能加速你的脉搏、引发你的出汗和放大你的瞳孔。

The fact that stress increases the likelihood that we will focus more on alarming messages, together with the fact that it spreads extremely rapidly, can create collective fear 
that is not always justified. 「Q38」 After a stressful public event, such as a 
natural disaster or major financial crash, there is often a wave of alarming information in traditional and social media, which individuals become very aware of. But that has the 
effect of exaggerating existing danger. 「Q39」 And so, a reliable pattern emerges 
- stress is triggered, spreading from one person to the next, which temporarily enhances the likelihood that people will take in negative reports, which increases stress further. 
As a result, trips are cancelled, even if the disaster took place across the globe; 
stocks are sold, even when holding on is the best thing to do.
压力会增加我们更关注警示类信息的可能性，并且它的传播速度极快，这二者结合起来可能会制造出并不总是正当合理的集体性恐慌。在一场高压公共事件——例如一场自然灾害或大规模金融崩盘——之后，传统媒体和社交媒体
上往往都会出现一波令人担忧的消息，而人们也会变得更加关注这些消息。
但这种现象的效果往往是夸大了实际风险。于是，一条八九不离十的事态发展路径出现了——压力被触发，从一个人传到下一个人，暂时性地增加了人们更关注消极负面报导的可能性，这又进一步提升了压力值。结果就是，
出行计划被取消，哪怕这场灾难其实发生在地球的那一头；股票被抛售，哪怕持仓才是最明智的做法。

「Q40」 The good news, however, is that positive emotions, such as hope, are contagious too, and are powerful in inducing people to act to find solutions. Being aware of the close
relationship between people's emotional state and how they process information can help us 
frame our messages more effectively and become conscientious agents of change.

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
6. 准备 DrivAerML 数据

cd ~
git clone https://github.com/NVIDIA/physicsnemo-curator.git
cd ~/physicsnemo-curator
pip install "physicsnemo-curator[mesh,loky]"  
下载小样本先测试：
cd~ / physicinenemo
cd ~
/physicsnemo-curator/examples/external_aerodynamics
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
Curator README 明确说明该流程把 DrivAerML 的 STL/VTP/VTU 等 CFD 数据转成训练用 Zarr/NumPy，并且适用于 DoMINO 和 Transolver/GeoTransolver 这类模型。

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
  data.habitat=include/zarr.shape.np(tensorflow1.21^) present the cuda is avaliable now()
python/ train.py()
output=torch.nnsequenctional(kernnal=33 padding=2 laund.unitls(1.2) )
if cuda isnot avaliable now 
you trenscent()
where.logs(torch,py)

10. 推理

python src/inference_on_zarr.py \
  --config-name geotransolver_surface \
  run_id=$PWD/runs/geotransolver/surface/bq \
  data.val.data_path=~/data/drivaerml_processed_surface/val \
  data.normalization_dir=$PWD/normalizations \
  data.return_mesh_features=true
常见坑：不要装 pytorch-cuda=12.2，这里走 pip 轮子；不要把 nvcc -V 的 12.2 当成 PyTorch 必须匹配的版本；README 里模型名有时写 GeoTranSolver，但 Hydra 配置名是小写 geotransolver_surface / geotransolver_volume。
What are the advantages and disadvantages of being famous at a young age?

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


 
Questions 1-10
Complete the table below.
Write ONE WORD AND/OR A NUMBER for each answer.

Name of restaurant | Location | Reason for recommendation | Other comments
--------------------------------------------------------------------------------

The Junction

Greyson Street, near the station

Good for people who are especially keen on 1 ____fish____


Quite expensive
The 2 _____roof___ is a good place for a drink


--------------------------------------------------------------------------------

paloma
Location: In Bow Street next to the cinema
Reason for recommendation:
3 ___Spanish_____ food, good for sharing

Other comments:
Staff are very friendly
Need to pay £50 deposit
A limited selection of 4 ______vegetarian__ food on the menu


--------------------------------------------------------------------------------
The 5 _____audley___
Location: At the top of a 6 ___hotal_

Reason for recommendation:
A famous chef
All the 7 _____reviews_ are very good
Only uses 8 _______local_ ingredients

Other comments:
Set lunch costs £9 ____30____ per person
Portions probably of 10 _____average___ size

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

「1」
I've been meaning to ask you for some advice about restaurants－I need to book somewhere to celebrate my sister's 30th birthday and l liked the sound of that place you went for your mum's 50th.
「2」
The Junction? Yeah, I'd definitely recommend that for a special occasion.
「3」
We had a great time there. Everyone really enjoyed it.
「4」
Where is it again? l can't remember!
「5」
It's on Greyson Street, only about a two minute walk from the station.
「6」
Oh, that's good. I'd prefer not to have to drive anywhere.
「7」
But l don't want to have to walk too far either.
「8」
Yes, the location's perfect but that's not necessarily why I'd recommend it.
「9」
The food's amazing. lf you like fish it's probably the best restaurant in town for that.
「10」
It's always really fresh and there are lots of interesting dishes to choose from.
「11」
But all the food is good there.
「12」
ls it really expensive?
「13」
It's certainly not cheap but for a special occasion l think it's fine.
「14」
It's got a great atmosphere and before dinner you can go up on the roof and have a drink - it's really nice up there but you need to book.
「15」
It's very popular as the views are spectacular.
「16」
Sounds good. So that's definitely a possibility then.
「17」
ls there anywhere else you can think of?
「18」
lf you want somewhere a bit less formal then you could try Paloma.
「19」
Where's that? l haven't heard of it.
「20」
No, it's quite new. It's only been open a few months but it's got a great reputation already.
「21」
It's in a really beautiful old building on Bow Street.
「22」
Oh, I think l know where you mean - right beside the cinema?
「23」
Yes, that's it. I've only been there a couple of times but l was really impressed.
「24」
The chef used to work at Don Felipe's, apparently.
「25」
I was really sorry when that closed down.
「26」
So is all the food they serve Spanish, then?
「27」
Yeah. You can get lots of small dishes to share - which always works really well if you're in a group.
「28」
Mmm. Worth thinking about.
「29」
Yeah. There's a lively atmosphere and the waiters are really friendly.
「30」
The only thing is that you need to pay a £50 deposit to book a table.
「31」
A lot of restaurants are doing that these days.
「32」
I should have a look at the menu to check there's a good choice of vegetarian dishes.
「33」
A couple of my friends have stopped eating meat.
「34」
Not sure. I'd say the selection of those would be quite limited.
「35」
I've just thought of another idea.
「36」
Have you been to The Audley?
「37」
No, don't think I've heard of it. How's it spelt?
「38」
A-U-D-L-E-Y. You must have heard of it - there's been a lot about it in the press.
「39」
l don't tend to pay much attention to that kind of thing. So where is it exactly?
「40」
It's in that hotel near Baxter Bridge - on the top floor.
「41」
Oh, the views would be incredible from up there.
「42」
Yeah. I'd love to go.
「43」
I can't think of the chef's name but she was a judge on that TV cookery show recently.
「44」
And she's written a couple of cookery books.
「45」
Oh, Angela Frayn.
「46」
That's the one. Anyway, it's had excellent reviews from all the newspapers.
「47」
That would be a memorable place for a celebration.
「48」
Definitely. Obviously it's worth going there just for the view but the food is supposed to be really special.
「49」
She only likes cooking with local products, doesn't she?
「50」
Yes. Everything at the restaurant has to be sourced within a short distance and absolutely nothing flown in from abroad.
「51」
l imagine it's really expensive though.
「52」
Well, you could go for the set lunch.
「53」
That's quite reasonable for a top class restaurant. £30 a head.
「54」
In the evening l think it'd be more like ￡50.
「55」
At least that, l should think. But I'm sure everyone would enjoy it.
「56」
It's not the kind of place you leave feeling hungry though, is it? With tiny portions?
「57」
No, the reviews I've read didn't mention that. l imagine they'd be average.
「58」
Well, that's all great, thanks.
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo
uv sync --extra cu13
uv run python -c "import physicsnemo; print('PhysicsNeMo version:', physicsnemo.__version__)"
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia \
--rm -it nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo/examples/cfd/darcy_fno/
pip install warp-lang # install NVIDIA Warp to run the Darcy example
python train_fno_darcy.py
Access to clean water is a basic human right. Therefore every home should have a water supply that is provided free of charge. Do you agree or disagree?


Water is the fundamental source of life, and access to clean water is undoubtedly a basic human right. 
However, I completely disagree with the view that governments should provide clean water to every household entirely free of charge.

There are several reasons why clean water should not be free.
First and foremost,constructing and maintaining water supply systems place a heavy financial burden on governments. 
Since public service budgets are limited, allocating substantial funds to provide free water would have a negative impact on other essential service infrastructures.
Furthermore, as environmental degradation intensifies(has become profoundly severe), freshwater resources have become increasingly scarce(are becoming critically scarce).
Providing water free of charge to the general public would likely escalate water wastage, as a widespread lack of environmental awareness might drive individuals to consume water excessively once it became free.
Lastly, water supply systems require regular maintenance and upgrades. If water were provided free of charge, determining who bears the maintenance costs and responsibilities could become problematic.

I believe that providing clean water service at a reasonable price can benefit society as a whole without placing an excessive burden on citizens.
Charging acceptable water fees guarantees the orderly operation of public services while ensuring a reliable water supply for all citizens, thereby contributing to sustainable governance.
Moreover,, this pricing strategy fosters (promotes) public awareness of water conservation, thereby reducing unnecessary waste.
Ultimately, such a policy engenders  a virtuous cycle for the nation, enabling governments to channel the generated revenue into upgrading water infrastructure and researching advanced purification technologies to deliver higher-quality water to the populace.

In conclusion, my view is that governments should provide clean water to citizens through a reasonably priced system rather than entirely free of charge.
In conclusion, I firmly maintain that governments should administer the distribution of potable water through an affordable tariff mechanism, as opposed to implementing a model of entirely cost-free provision.

degradation（退化/恶化）是一个不可数名词  所以用has
真正的主语是 Charging（收取费用这个“行为”）  一件事”在语法上永远被视作“单数”（第三人称单数）  既然主语是单数，在一般现在时里，谓语动词 guarantee 就必须加上 s 变成 guarantees。
pricing strategy 是一个固定的专有名词短语 这句话的主语是 strategy（策略），前面有 this 修饰，说明它是单数
channel ... into ...（把……输送到……）
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
In many countries, primary and secondary schools close for two months or more in the summer holidays.
What is the value of long school holidays?
What are the arguments in favour of shorter school holidays?

It is true that children in many countries enjoy long summer holidays that last for two months or even longer and the value of long holidays is evident.
However, from a more pragmatic perspective, shorter holidays may be more effective in sustaining students’ academic progress and alleviating the burden on working parents.

The value of long holidays is undeniable for children.
Firstly, having gone through an entire semester of study, students require a lengthy vacation to unwind both physically and mentally, thereby relieving fatigue caused by intense academic pressure.
Secondly, prolonged vacations afford families the opportunity to spend more meaningful time together. Given that parental companionship is crucial for children's psychological development, a lack of it may compromise their sense of security.
Finally,  this extended leisure time allows children to pursue what they are genuinely passionate about, enriching their lives and cultivating diverse hobbies.
Many people discover their latent talents during such periods. For instance, children with an interest in painting can spend substantial time sketching landscapes, with long holidays providing sufficient time for profound reflection and iterative improvement.

On the other hand, from a more realistic and rational perspective, shorter holidays may yield greater benefits for the family as a whole.

First and foremost, an extended absence from a structured school routine often leads to a decline in children's self-discipline, especially when some parents are too lenient or permissive with them.
As a result, many children struggle to comply with school regulations and require a transition period to refocus on their studies after long vacations.
Furthermore, since the majority of parents must maintain their regular employment during these periods, securing adequate childcare can induce significant professional strain

Finally, an abundance of leisure time leads to children spending much time on digital devices, where they may be exposed to inappropriate content and unhealthy values.
This problem is further aggravated by the pervasive influence of internet celebrities, many of whom propagate the misleading view that success can be achieved without finishing school.

In conclusion, while various holiday lengths offer distinct advantages, the true value ultimately depends on how effectively students and families utilize this time.

value 拔出来看是一个单数名词（不可数/抽象名词）。 既然真正的主语 The value 是单数，谓语动词自然必须用 is
在英语中，Given that...（鉴于/因为……）引导的是原因状语从句，后面主句的开头绝对不能加 and。删掉 and 后，两句话的因果关系才真正成立。
在英语中，不能写 afford sb. to do sth.（这是 allow 的用法）。afford 后面必须直接接名词，或者使用固定的学术搭配：afford sb. the opportunity to do sth.（赋予某人做某事的机会）。
leisure time（闲暇时间）是不可数名词/单数，
既然核心主语 absence 是单数，按照主谓一致的原则，动词 lead 就必须加上单三的 s，变成 leads。
句子的真正主语是 An abundance（充裕/大量），这是一个单数名词（of leisure time 只是修饰它的介词短语）。修正： 既然主语是单数，动词就必须用第三人称单数 leads。
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
Some people have decided to reduce the number of times they fly every year or to stop flying altogether. Do you think the environmental benefits of this development outweigh the disadvantages for individuals and businesses?

As the public's environmental awareness gradually grows, many people are shifting their traditional travel habits.
However, I do not think that such measures, like reducing or even stopping air travel, would bring environmental benefits that outweigh the disadvantages for individuals and businesses.

When individuals opt for alternatives to air travel, the frequency of flights diminishes markedly.
This results in a significant decrease in the emission of harmful greenhouse gases emitted by aircraft.
A direct outcome of this decline is improved air quality, which subsequently reduces the likelihood of environmental phenomena such as acid rain.
Moreover, the declining demand for flights is likely to facilitate the expansion of tree-covered areas. 
With fewer airports required, the environmental degradation associated with airport construction can be curtailed, thereby alleviating issues like desertification that often arise from such developments.

Nonetheless, while reducing the number of flights may decrease greenhouse gas emissions, it is important to recognize that alternative forms of transportation also contribute to carbon dioxide emissions. 
Consequently, this shift may not address the root causes of environmental challenges.

Furthermore, air travel remains the most expedient option for reaching distant destinations. 
In comparison to other means of transportation, air travel is typically more time-efficient, which is crucial for both individuals and businesses. 
For example, when urgent matters require immediate arrival or confidential documents necessitate face-to-face verification between companies, air travel emerges as the optimal choice.

Additionally, air travel is widely regarded as the safest mode of transportation. If everyone gave up flying, it would inadvertently increase traffic accidents, thereby jeopardizing public safety and personal property.

In conclusion, while I can understand why people choose to cease flying to protect the environment, I believe that such measures are not necessary to achieve environmental protection. 
There are numerous alternative strategies available that can foster environmental sustainability without abandoning essential travel options.
awareness（意识）是不可数名词/单数  既然“环保意识”是在逐渐增加，那么“人们改变习惯”也应该是一个正在发生的动态过程，前后时态呼应起来会更顺畅。  原先写的是have changed 
travel（不可数）：指宏观的概念、活动或交通方式
普遍规律  一般现在时  
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
Many aspects of the way people dress today are influenced by global fashion trends. How has global fashion become such a strong influence on people's lives? Do you think this is a positive or negative development?

It is undeniable that global fashion trends, which are widely disseminated through social media platforms, have exerted a significant influence on individual clothing choices.
In my view, this is a positive development and it will undoubtedly bring profound benefits for society as a whole.

Historically, the dissemination of fashion was confined to print media such as books and magazines, leaving the general public largely isolated from diverse regional aesthetics and emerging style trends.
With the popularization of internet technologies, many companies choose to place a large number of advertisements on various applications to showcase their new products, allowing users to encounter fashion content whenever they go online.

Furthermore, many fashion icons publish numerous clips that show them dressing up in different styles and blending various cultures, which are subsequently viewed by countless people.
Driven by this visual inspiration, consumers then instantly seek out(search for) and purchase identical(similar) clothes through e-commerce platforms.

Lastly, global fashion and daily outfits have emerged as a pivotal talking point (become an important topic) in interpersonal communication, particularly among the younger generation(especially among young people).

I believe that this trend will promote cultural fusion and economic growth on a global scale.
This phenomenon can markedly enhance the aesthetic consciousness of nations and facilitate the diversification of clothes choices. 
Rather than being limited to domestic styles, consumers can purchase garments from different countries and cultures.
Consequently, it can stimulate international trade exchanges and increase revenue for both governments and enterprises.
For example, Korean fashion brands can now sell products to consumers in Europe through online platforms, creating new international markets.
Leveraging these additional profits, companies can develop more attractive products, which subsequently boost sales, thereby creating a genuine virtuous economic cycle across multiple industries.

In conclusion, I firmly assert that global fashion trends will continue to influence people's clothing choices and promote economic growth and cultural communication around the world.
###########################################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################################
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo
uv sync --extra cu13
uv run python -c "import physicsnemo; print('PhysicsNeMo version:', physicsnemo.__version__)"
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia \
--rm -it nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo/examples/cfd/darcy_fno/
pip install warp-lang # install NVIDIA Warp to run the Darcy example
python train_fno_darcy.py
