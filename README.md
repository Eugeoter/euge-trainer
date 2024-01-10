一个魔改 kohya-ss/sd-scripts 的 SDXL 微调的项目，原项目地址：https://github.com/kohya-ss/sd-scripts

## 特性

### 1. 优化

在 kohya-ss 的基础上，对训练脚本进行了优化，主要包括：

1. 重写了数据集类：
   a. 实现了数据库读取，从 json 文件中读取需要的所有参数；
   b. 实现了多 GPU 并行缓存潜变量；
   c. 实现了异步潜变量缓存写入，通过异步 IO 来提高缓存潜变量时的 GPU 利用率，从而加速缓存；
   d. 实现了缓存文件有效性检查；
   e. 支持多线程数据加载；
   f. 支持仅使用缓存文件，而不需要图像文件的训练；
   g. 支持非一次性的缓存读取，而是仅在开始时检查缓存的有效性，随训练逐步读取，降低了从运行脚本到开始训练的等待时间；
2. 调整了标注处理算法；
3. 实现了自动重复次数计算：自动根据不同数据的特征确定其在一个 epoch 内的重复次数；
4. 支持单独缓存潜变量：仅加载 VAE 来缓存潜变量，而非在训练开始时加载所有组件后再缓存。
5. 优化了控制台日志：
   a. 在训练进度条中添加了更多的训练信息，如学习率、当前 loss、平均 loss、到下一个 epoch 需要的步数等；
   b. 在多 GPU 训练时避免重复打印冗余信息；
   c. 训练前打印了更多训练参数；
6. 新增了模型保护机制：在训练出错/人为打断训练时自动保存模型文件，防止训练进度丢失。

### 2. 不支持的功能

1. 不支持 `{图像文件名}.txt` 的标注文件，而直接使用元数据 json 文件；
2. 不支持训练期间的预览图生成：麻烦，似乎预览图的参考价值不高；

## 使用方法

### 1. 安装

`git clone https://github.com/Eugeoter/sdxl-trainer`
`cd sdxl-trainer`
`pip install -r requirements.txt`

### 2. 训练

#### 2.1. 数据

准备好图像（或潜变量缓存）文件和元数据 json 文件，将图像（或潜变量缓存）文件放置在同一文件夹 image_dir 内。

#### 2.2. 参数

可在 params.txt 内配置参数后，复制到控制台终端执行，具体参数含义见 modules/arg_utils.py

#### 2.3. 多 GPU 训练

将原本参数中的 `accelerate launch sdxl_train.py` 更换为 `accelerate launch --num_processes=4 --multi_gpu --gpu_ids=0,1,2,3 sdxl_train.py` 即可进行多 GPU 训练，其中 `--num_processes` 为进程数，`--gpu_ids` 为 GPU 编号。

#### 2.4. 缓存潜变量

可使用相同参数执行 `accelerate launch cache_latents.py` 缓存潜变量，缓存后的潜变量文件将被储存为 `{图像文件名}.npz` 的格式。
