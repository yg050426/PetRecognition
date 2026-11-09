# 基于深度神经网络的宠物识别系统设计与实现

本科毕业设计项目 ｜ 品种识别 + 鼻纹个体识别 + 端侧量化部署

---

## 这个项目在做什么

一套**完全在本地运行**的宠物识别系统，同时解决两个问题：

- **品种识别**：输入一张照片，输出 37 类品种标签与置信度（闭集分类）
- **个体识别**：用鼻纹与面部特征确认"是哪一只"（开集比对），用于走失寻回

现有方案要么依赖植入芯片（有创伤、需专用设备），要么依赖云端接口（封闭、需联网）。本课题的目标是把这两件事做进一个可离线运行的桌面/移动客户端。

## 技术路线

`数据 → 算法 → 压缩 → 系统 → 验证` 五段贯通

| 环节 | 做法 |
| --- | --- |
| 数据 | Oxford-IIIT Pet（37 类）＋ Stanford Dogs（120 类）；个体识别数据自采，Labelme 标注鼻部/面部区域 |
| 算法 | 骨干 ResNet18 / ResNet50 / EfficientNet-B0，ImageNet 预训练权重迁移学习 + 分阶段微调；个体识别走"先检测后识别"：YOLOv8n 定位裁剪 → ArcFace 加性角度间隔损失训练 512 维特征向量 |
| 压缩 | 知识蒸馏（ResNet50 → MobileNetV3）、INT8 量化、结构化剪枝，统一导出 ONNX |
| 系统 | Qt 6 + C++17，界面/业务/推理/持久四层，SQLite 存档案与特征向量；桌面端经 ONNX Runtime C++ 接口推理 |
| 验证 | 功能测试用例 + 性能测试报告（模型体积、单张推理时延、宏平均 P/R/F1、混淆矩阵） |

## 当前进度（诚实标注）

| 项 | 状态 |
| --- | --- |
| 实验环境（RTX 4060 Laptop 8GB + PyTorch 2.7.1 + CUDA 11.8） | ✅ 已完成 |
| 数据集下载、解包为 ImageFolder、完整性校验 | ✅ 已完成（Pet 3680 张训练 / 3669 张测试；Dogs 14406 / 6174；坏图 0） |
| 分层切分验证集（15%，清单式不搬原图） | ✅ 已完成 |
| 预训练模型推理通路验证 | ✅ 已完成 |
| ResNet18 品种识别训练（frozen / finetune） | ✅ 已完成，结果见下 |
| 官方测试集评测脚本 | ✅ 已完成 |
| ResNet50 / EfficientNet-B0 骨干对比 | ⏳ 进行中 |
| SE 注意力、Focal Loss、数据增强消融 | ⏳ 未开始 |
| 蒸馏 / INT8 量化 / 剪枝 | ⏳ 未开始 |
| ArcFace 个体识别 | ⏳ 未开始（自采数据渠道待确认） |
| Qt6 + C++17 桌面客户端 | ⏳ 未开始 |

## 实测结果

在 **Oxford-IIIT Pet 官方测试集（3669 张，37 类）** 上：

| 配置 | 验证集 Top-1 | **官方测试集 Top-1** | Top-5 | 宏平均 F1 | 每轮耗时 | 显存峰值 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 frozen（仅训 fc，3 轮） | 86.2% | **84.76%** | 98.45% | 85.40% | 22 s | 0.29 GB |
| ResNet18 finetune（全解冻，20 轮，best = ep08） | 90.7% | **87.33%** | 98.47% | 87.80% | 28 s | 0.84 GB |

**与目标的差距**：课题承诺品种识别 Top-1 不低于 90%，当前官方测试集实测 **87.33%**，差 2.67 个百分点，**尚未达标**。

**诊断**：finetune 从第 11 轮起 `train_top1 = 1.000`（训练集 3129 张完全饱和），而 `val_top1` 在第 8 轮到顶后回落 —— 当前瓶颈是**正则强度不足**，不是模型容量不够。下一步：加强数据增强 + Dropout + 权重衰减，并对比 ResNet50 与更高分辨率输入。

> 说明：表中"验证集"是从 train 里分层切出的 551 张，用于调参与选模型；"官方测试集"3669 张全程不参与训练与调参。两者口径不同，不可混用。

## 仓库结构

```
split_val.py          分层切验证集（生成 manifests 清单，不搬动任何图片）
train.py              两阶段迁移学习训练（frozen / finetune），带 CSV 实验日志与 best 权重保存
eval_test.py          官方测试集评测：Top-1 / Top-5 / 宏平均 P-R-F1 / 逐类精度 / 混淆矩阵
try_infer.py          环境与预训练推理通路验证
download_data.py      数据集下载（走 hf-mirror parquet 分片）
unpack_datasets.py    解包为 ImageFolder 结构，类别目录统一补零
scripts/              数据规范化（目录名补零、按磁盘重建 labels、导出 class_synsets）
logs/                 实验记录 CSV（训练过程 + 测试集评测 + 混淆矩阵）
```

`data/`、`checkpoints/`、`manifests/` 体积大且可再生，已在 `.gitignore` 中排除。

## 怎么跑

```bash
conda create -n pet python=3.10 && conda activate pet
pip install torch torchvision   # 需 CUDA 版本请按官网命令安装

python download_data.py                 # 下载数据集
python unpack_datasets.py               # 解包为 ImageFolder
python scripts/fix_dogs_names.py        # 类别目录统一补零
python split_val.py                     # 分层切 15% 验证集
python train.py --dataset pet --arch resnet18 --stage frozen --epochs 10
python train.py --dataset pet --arch resnet18 --stage finetune --epochs 20 --lr 1e-4
python eval_test.py --ckpt checkpoints/pet_resnet18_finetune_best.pth --arch resnet18
python eval_test.py --ckpt checkpoints/pet_resnet50_finetune_best.pth --arch resnet50 --split val
```

`--split test`（默认）读 `data/pet/test` 目录，是承诺指标的口径；
`--split val` 读 `manifests/pet_val.txt` 清单——验证集不是独立目录，是 `split_val.py` 从 train 里分层挑出来的 551 张，只有名单没有文件夹。

## 实验留图

每轮训练 / 评测跑完，**先出图，再写进展**：

```bash
python make_figs.py --ckpt checkpoints/pet_resnet50_finetune_best.pth --arch resnet50
python log2fig.py logs/run_resnet50.log
powershell -ExecutionPolicy Bypass -File capture.ps1 -Tag ui_main
```

- `make_figs.py` 从 `logs/*.csv` 直接画论文用的成图（300 dpi，宽 15 cm），
  图号 / 图注 / 数据来自哪个文件 / 那个文件几点写的，自动记进 `figures/图注清单.md`。
- `log2fig.py` 把终端日志转成带时间戳和 SHA1 的证据图，放 `figures/证据/`。
- `capture.ps1` 一键截屏，给软件界面、目录结构、GPU 占用这类现场证据用。

完整规矩见 `留图规范.md`。`figures/` 不进仓库（里面是数据集原图）。

## 推送前必查（公开仓库）

```bash
python scripts/leak_check.py --all     # 0 命中才允许 push
```

扫的是**已跟踪 + 未跟踪且未被忽略**的全部文本文件：姓名、学校、学院、班级、
教师姓名、学号、手机号、QQ 邮箱、带用户名的本机绝对路径。
词表放在 `scripts/leak_patterns.txt`，**已 gitignore**——扫描器自身不能含要查的词，
否则它自己就成了新的泄露源。

历史改写过之后，验证方式只有一个可信：**从 GitHub 重新克隆一份，再扫全部对象和提交说明**。
只看当前 HEAD、只扫本地工作区都不够。

## 数据集出处

- Parkhi O M, Vedaldi A, Zisserman A. Cats and dogs[C]. CVPR 2012.（Oxford-IIIT Pet）
- Khosla A, Jayadevaprakash N, Yao B. Novel dataset for fine-grained image categorization: Stanford dogs[C]. CVPR Workshop 2011.
