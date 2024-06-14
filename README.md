# Euge Trainer

一个魔改 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 的 SD 微调的项目。

# 特性

在 kohya-ss 的基础上，对训练脚本进行了优化，主要包括：

1. **重写了数据集类**：
   a. 实现了数据库读取，从 json/csv/sqlite3 文件中读取需要的所有参数；
   b. 实现了多 GPU 并行缓存潜变量；
   c. 实现了异步潜变量缓存写入，通过异步 IO 来提高缓存潜变量时的 GPU 利用率，从而加速缓存；
   d. 实现了缓存文件有效性检查；
   e. 支持多线程数据加载；
   f. 支持仅使用缓存文件，而不需要图像文件的训练；
   g. 支持非一次性的缓存读取，而是仅在开始时检查缓存的有效性，随训练逐步读取，降低了从运行脚本到开始训练的等待时间；
2. **支持单独缓存潜变量**：仅加载 VAE 来缓存潜变量，而非在训练开始时加载所有组件后再缓存。
3. **更美观的控制台日志**
4. **更安全的模型保护机制**：在训练出错/人为打断训练时自动保存模型文件，防止训练进度丢失。
5. **更灵活的配置文件**：使用配置文件来配置训练参数，支持多种自定义参数。
6. **更自由的训练策略定制**：提供训练策略自定义方式，支持自定义训练策略，包括自定义每个数据的重复次数、自定义标注处理方式等。

# 使用方法

## 安装

```bash
git clone https://github.com/Eugeoter/euge-trainer.git
cd euge-trainer
git checkout dev
pip install -r requirements.txt
```

## 训练

### 准备数据

#### 本地数据集

准备好图像（或潜变量缓存）文件和同名的标注文件（或整一个元数据 json 文件），将图像（或潜变量缓存）文件放置在同一文件夹 image_dir 内。

训练器会加载配置中 `image_dirs` 和 `metadata_files` 所指向的数据。其中，`image_dirs` 是一个字符串列表，包含了所有的图像文件夹路径。`metadata_files` 是一个字符串列表，包含了所有的元数据文件路径。

数据加载方式有两种，一种是加载图像和与图像同名的 txt 文本文件作为标注，另一种是直接从元数据文件中加载标注。具体使用哪一种方式取决于 `metadata_files` 是否为空。

- 当 `metadata_files` 为空时，训练器会加载 `image_dirs` 中的所有图像文件，并读取与图像同名的 txt 文件作为标注。
- 当 `metadata_files` 不为空时，训练器会从 `metadata_files` 中的元数据文件中读取每个数据的信息，包括了图像路径、标注、原始尺寸等等。该方式无需为每个图像文件创建一个 txt 文件作为标注，但需要额外的元数据文件。
  当某个元数据中的图像路径不存在时，训练器会尝试从 `image_dirs` 中查找文件名（不包含路径）相同的图像文件。

无论以哪种加载数据，请确保所加载的数据中没有去除后缀后重名的图像文件，否则会导致数据加载错误。

您可以直接使用由缓存潜变量生成的 npz 缓存文件代替原始图像文件。

以下几种数据组织方式均可：

##### a. 图像 + 图像同名 txt 标注文件：

```
  image_dir
  ├── image1.jpg
  ├── image1.txt
  ├── image2.jpg
  ├── image2.txt
  ├── ...
```

##### b. 缓存潜变量 + 潜变量同名 txt 标注文件：

```
  latent_dir
  ├── image1.npz
  ├── image1.txt
  ├── image2.npz
  ├── image2.txt
  ├── ...
```

##### 元数据文件

元数据文件是一个 json, csv 或 sqlite3 文件，包含了一些数据的各种信息，如图像路径、标注、图像尺寸、美学评分等。训练时，可以用一个或多个元数据文件来描述数据集。

使用元数据文件替代传统 txt 标注能简化数据集管理，不再需要为每个图像文件创建一个 txt 文件作为标注。其能够存储更多种类的数据信息。

制作元数据文件可使用 [Waifuset](https://github.com/Eugeoter/waifuset) 项目的 dev 分支。是时候摆脱 txt 标注文件了。

元数据文件格式：

**JSON**

```json
{
	"image1": {
			"image_path": ".../image1.jpg", // 图像路径
			"caption": "1girl, solo, ...", // 标注
      ... // 其他元数据
    },
  "image2": {
      "image_path": ".../image2.jpg",
      "caption": "1girl, solo, ...",
      ...
    },
  ...
}
```

**CSV**

```csv
image_path,caption
.../image1.jpg,1girl, solo, ...
.../image2.jpg,1girl, solo, ...
...
```

**SQL**

```sql
CREATE TABLE images (
  image_path TEXT PRIMARY KEY,
  caption TEXT
);
```

##### c. 图像 + 元数据文件：

```
  image_dir
  ├── image1.jpg
  ├── image2.jpg
  ├── ...
  metadata.json / metadata.csv / metadata.sqlite3
```

##### d. 缓存潜变量 + 元数据 json 文件：

```
  latent_dir
  ├── image1.npz
  ├── image2.npz
  ├── ...
  metadata.json / metadata.csv / metadata.sqlite3
```

#### HuggingFace 数据集

您可以使用 HuggingFace 数据集。当 `metadata_files` 和 `image_dirs` 为空时，训练器会尝试从 `dataset_name_or_path` 指定的 HuggingFace 数据集中加载数据，所加载数据的 `image` 列和 `caption` 列将被用作图像和标注。
图像列由参数 `dataset_image_column` 指定，标注列由参数 `dataset_caption_column` 指定。

### 配置参数

在 [configs/train_config.py](configs/config_train_sd3.py) 内配置参数（或自行新建）后，执行 `accelerate launch train_sd3.py --config configs/config_train_sd3.py` 即可开始训练。

其中，`--config` 参数为配置文件路径，`configs/config_train_sd3.py` 为默认配置文件，您需要根据自己的需求修改配置文件，或指定您自己修改的配置文件路径。

完整的参数介绍请参考 [docs/CONFIG.md](docs/CONFIG.md)。

### 多 GPU 训练

将原本参数中的 `accelerate launch train_sd3.py --config configs/config_train_sd3.py` 更换为 `accelerate launch --num_processes=4 --multi_gpu --gpu_ids=0,1,2,3 train_sd3.py --config configs/config_train_sd3.py` 即可进行多 GPU 训练，其中 `--num_processes` 为进程数，`--gpu_ids` 为 GPU 编号。
