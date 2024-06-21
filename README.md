# Euge Trainer

一个魔改 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 的 SD 微调的项目。

# 特性

在 kohya-ss 的基础上，对训练脚本进行了优化。支持 HuggingFace 数据集，支持多 GPU 训练，支持元数据文件，支持缓存潜变量。

# 使用方法

## 安装

```bash
git clone https://github.com/Eugeoter/euge-trainer.git
cd euge-trainer
git checkout dev
pip install -r requirements.txt
```

## 快速开始

### 准备数据

数据集由两部分组成：数据和元数据。二者分开放置和读取。
数据即训练图像，元数据则是数据的详细信息，包括每个数据的标注等信息。

### 配置参数

在 [configs/train_config.py](configs/config_train_sd3.py) 内配置参数（或自行新建）后，执行 `accelerate launch train_sd3.py --config configs/config_train_sd3.py` 即可开始训练。

其中，`--config` 参数为配置文件路径，`configs/config_train_sd3.py` 为默认配置文件，您需要根据自己的需求修改配置文件，或指定您自己修改的配置文件路径。

完整的参数介绍请参考 [docs/CONFIG.md](docs/CONFIG.md)。

### 多 GPU 训练

将原本参数中的 `accelerate launch train_sd3.py --config configs/config_train_sd3.py` 更换为 `accelerate launch --num_processes=4 --multi_gpu --gpu_ids=0,1,2,3 train_sd3.py --config configs/config_train_sd3.py` 即可进行多 GPU 训练，其中 `--num_processes` 为进程数，`--gpu_ids` 为 GPU 编号。
