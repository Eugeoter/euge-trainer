# Euge Trainer

一个魔改 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 的 SD 微调的项目。

# 特性

在 kohya-ss 的基础上，对训练脚本进行了优化。支持 HuggingFace 数据集，支持多 GPU 训练，支持元数据文件，支持缓存潜变量。
# 更新
加入了对tile模型的训练支持，下面是tile模型的训练方式


启动config.control_image_getter参数

并将config.control_image_type参数注释掉
```

#config.control_image_type = "canny"

config.control_image_getter = get_random_tile_condition
```

可以在己有ControlNet模型上继续训练

使用controlnet_model_name_or_path参数即可



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

一个简单的数据集由两部分组成：图片文件和标注文件，标注文件放在用于图片文件同名的 txt 文本文件中，例如：

```
data/
|-- 0001.jpg # 图片文件
|-- 0001.txt # 标注文件，例如，"a dog playing with a ball"
|-- 0002.jpg
|-- 0002.txt
|-- ...
```

### 配置参数

在 [configs/config_train_sdxl.py](configs/config_train_sdxl.py) 内配置参数（或自行新建）后，执行 `accelerate launch train.py --trainer sdxl --config configs/config_train_sdxl.py` 即可开始训练。

其中，`--config` 参数为配置文件路径，`configs/config_train_sdxl.py` 为默认配置文件，您需要根据自己的需求修改配置文件，或指定您自己修改的配置文件路径。

完整的参数介绍请参考 [docs/config.md](docs/config.md)。



## 解压arrow文件
import os
import tqdm
```
arrow_file = r"C:\Users\xx\Downloads\00000.arrow"
save_dir = r"D:\data\openpose"
```
save_image_dir = os.path.join(save_dir, 'image')
save_condition_image_dir = os.path.join(save_dir, 'condition_image')
os.makedirs(save_image_dir, exist_ok=True)
os.makedirs(save_condition_image_dir, exist_ok=True)

with pa.OSFile(arrow_file, 'rb') as source:
    with pa.RecordBatchFileReader(source) as reader:
        table = reader.read_all()

        for i, row in tqdm.tqdm(table.to_pandas().iterrows()):
            image = row['image']
            condition_image = row['condition_image']
            meta_info = row['meta_info']
            image_key = 'danbooru_' + str(meta_info['danbooru_pid'])
            caption = meta_info['caption_base'].replace('|||', '')
            with open(os.path.join(save_image_dir, image_key + '.png'), 'wb') as f:
                f.write(image)
            with open(os.path.join(save_condition_image_dir, image_key + '.png'), 'wb') as f:
                f.write(condition_image)
            with open(os.path.join(save_image_dir, image_key + '.txt'), 'w') as f:
                f.write(caption)

print(f"完成，共处理了{len(table)}张图片")
## 配置文件夹加入预处理文件的路径
```
config.dataset_source = [
    dict(
        name_or_path=r"图片文件夹路径",
        read_attrs=True,
    ),
    dict(
        name_or_path=r"控制图片文件夹路径",
        column_mapping={
            'image_path': 'control_image_path'
        }
    )
]
```
    
