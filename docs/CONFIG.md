# 配置文件 Wiki

# 总览

| 配置项                            | 译名                       | 类型     | 必须修改 | 简介                                                                                         |
| --------------------------------- | -------------------------- | -------- | -------- | -------------------------------------------------------------------------------------------- |
| **pretrained_model_name_or_path** | 预训练模型路径             | str      | 是       | 指向一个 safetensors 的大模型文件                                                            |
| **image_dirs**                    | 图像文件夹路径列表         | list     | 是       |                                                                                              |
| **metadata_files**                | 元数据文件路径列表         | list     | 是       |                                                                                              |
| **output_dir**                    | 项目输出文件夹路径         | str      | 是       |                                                                                              |
| vae                               | VAE 模型路径               | str      | 否       | 指向一个 safetensors 的 vae 模型文件。将覆盖大模型自带的 vae。                               |
| no_half_vae                       | 不使用半精度训练 VAE       | bool     | 否       | 见[VAE 精度](#vae-精度)                                                                      |
| tokenizer_cache_dir               | 分词器缓存路径             | str      | 否       |                                                                                              |
| flip_aug                          | 是否使用水平翻转数据增强   | bool     | 否       | 启用时，训练图像会随机水平翻转。但缓存潜变量的时长和大小也会加倍。                           |
| bucket_reso_step                  | 分桶分辨率步长             | int      | 否       | 分桶图像的分辨率间隔，以 32 或 64 最佳。                                                     |
| resolution                        | 图像分辨率                 | int      | 否       | 分桶的最大分辨率，SDXL 通常为 1024。                                                         |
| max_dataset_n_workers             | 数据集的最大工作线程数     | int      | 否       | 若无特殊需求，使用 1 即可。                                                                  |
| max_dataloader_n_workers          | 数据加载器的最大工作线程数 | int      | 否       |                                                                                              |
| persistent_data_loader_workers    | 数据加载器工作线程持久化   | bool     | 否       |                                                                                              |
| loss_recorder_kwargs              | 损失记录器参数             | cfg      | 否       |                                                                                              |
| loss_recorder_kwargs.gamma        | 损失记录器的遗忘因子       | float    | 否       | 0~1 之间。数值越大越接近当前 loss。                                                          |
| loss_recorder_kwargs.stride       | 损失记录器记录的步幅       | int      | 否       | 记录最近平均 loss 时的步长。越小越接近当前 loss                                              |
| save_precision                    | 保存精度                   | str      | 否       | 通常与训练的混合精度相同。                                                                   |
| save_every_n_epochs               | 每 n 轮保存一次模型        | int      | 否       | 为 None 时不启用                                                                             |
| save_every_n_steps                | 每 n 步保存一次模型        | int      | 否       | 为 None 时不启用                                                                             |
| save_on_train_end                 | 训练结束时自动保存模型     | bool     | 否       |                                                                                              |
| save_on_keyboard_interrupt        | 键盘中断时自动保存模型     | bool     | 否       |                                                                                              |
| save_on_exception                 | 训练异常时自动保存模型     | bool     | 否       |                                                                                              |
| num_train_epochs                  | 训练总轮数                 | int      | 否       |                                                                                              |
| batch_size                        | 单卡批量大小               | int      | 否       |                                                                                              |
| learning_rate                     | 学习率                     | float    | 否       |                                                                                              |
| block_lr                          | 分块学习率                 | float    | 否       | 暂时弃用                                                                                     |
| lr_scheduler                      | 学习率调度器               | str      | 否       | 见[学习率和优化器](#学习率和优化器)。                                                        |
| lr_warmup_steps                   | 学习率预热步数             | int      | 否       | 见[学习率和优化器](#学习率和优化器)。                                                        |
| lr_scheduler_power                | 多项式学习率调度器的阶数   | float    | 否       | 见[学习率和优化器](#学习率和优化器)。                                                        |
| lr_scheduler_num_cycles           | 学习率调度器周期           | int      | 否       | 见[学习率和优化器](#学习率和优化器)。                                                        |
| lr_scheduler_kwargs               | 学习率调度器参数           | cfg      | 否       | 见[学习率和优化器](#学习率和优化器)。                                                        |
| full_bf16                         | 全 bf16 精度               | bool     | 否       | 启用时，以显存换精度。开启后，文本编码器将变为 bf16 精度。                                   |
| full_fp16                         | 全 fp16 精度               | bool     | 否       | 启用时，以显存换精度。开启后，文本编码器将变为 fp16 精度。                                   |
| train_text_encoder                | 训练文本编码器             | bool     | 否       |                                                                                              |
| learning_rate_te1                 | 文本编码器 1 学习率        | float    | 否       |                                                                                              |
| learning_rate_te2                 | 文本编码器 2 学习率        | float    | 否       |                                                                                              |
| gradient_checkpointing            | 梯度检查点                 | bool     | 否       | 启用时，以时间换显存。                                                                       |
| gradient_accumulation_steps       | 梯度累积步数               | int      | 否       | 在低显存下模拟高显存时的批量大小。                                                           |
| optimizer_type                    | 优化器类型                 | str      | 否       | 见[学习率和优化器](#学习率和优化器)。                                                        |
| optimizer_kwargs                  | 优化器参数                 | dict     | 否       | 见[学习率和优化器](#学习率和优化器)。                                                        |
| mixed_precision                   | 混合精度                   | str      | 否       | 为 None 时，使用全精度训练。详见混合精度介绍。                                               |
| cpu                               | 是否使用 CPU 训练          | bool     | 否       | 启用时，使用 cpu 代替 gpu 训练。极慢。                                                       |
| max_token_length                  | 最大 token 长度            | int      | 否       | 分词器的超参数，通常无需修改。                                                               |
| mem_eff_attn                      | 使用内存效率注意力机制     | bool     | 否       |                                                                                              |
| xformers                          | 使用 xformers              | bool     | 否       | 启用时，以轻微质量效果为代价，大大加速训练并降低显存占用。                                   |
| diffusers_xformers                | 使用 diffusers xformers    | bool     | 否       |                                                                                              |
| sdpa                              | 使用 sdpa                  | bool     | 否       |                                                                                              |
| clip_skip                         | 裁剪跳过次数               | int      | 否       | 通常为 1 或 2。                                                                              |
| noise_offset                      | 噪声偏移量                 | float    | 否       | 偏移初始噪声以帮助模型生成很暗或很亮的图像。                                                 |
| multires_noise_iterations         | 多分辨率噪声迭代次数       | int      | 否       |                                                                                              |
| multires_noise_discount           | 多分辨率噪声折扣           | float    | 否       |                                                                                              |
| adaptive_noise_scale              | 自适应噪声缩放             | float    | 否       |                                                                                              |
| max_grad_norm                     | 最大梯度                   | float    | 否       |                                                                                              |
| zero_terminal_snr                 | 零终端信噪比               | bool     | 否       | 启用时，强制归零时间步的信噪比。提高对数据集的还原度。                                       |
| ip_noise_gamma                    | 扰动噪声伽马               | float    | 否       |                                                                                              |
| min_snr_gamma                     | 最小信噪比伽马             | float    | 否       | 限制模型低时间步下的学习能力。加快拟合速度的同时防止死磕小细节。数值越低效果越强。           |
| scale_v_pred_loss_like_noise_pred | V 预测样本预测损失缩放     | bool     | 否       |                                                                                              |
| v_pred_like_loss                  | V 预测样本预测损失         | float    | 否       |                                                                                              |
| debiased_estimation_loss          | 去偏估计损失               | bool     | 否       |                                                                                              |
| min_timestep                      | 最小时间步                 | int      | 否       | 非专业人士勿动。                                                                             |
| max_timestep                      | 最大时间步                 | int      | 否       | 非专业人士勿动。                                                                             |
| timestep_sampler_type             | 时间步采样类型             | str      | 否       | 见[时间步采样介绍](#时间步采样)。                                                            |
| timestep_sampler_kwargs           | 时间步采样参数             | dict     | 否       | 见[时间步采样介绍](#时间步采样)。                                                            |
| sample_benchmark                  | 采样基准文件               | str      | 否       | 指向一个 json 文件，记录了采样的参数。                                                       |
| sample_every_n_epochs             | 每 n 轮采样一次            | int      | 否       |                                                                                              |
| sample_every_n_steps              | 每 n 步采样一次            | int      | 否       |                                                                                              |
| sample_sampler                    | 采样器                     | str      | 否       |                                                                                              |
| sample_params                     | 采样参数                   | dict     | 否       | 基础采样参数。用其中的参数填补采样基准各组参数缺失的参数。                                   |
| num_repeats_getter                | 重复次数获取器             | callable | 否       | 自定义功能。为 None 时禁用。见重复次数获取器介绍。                                           |
| caption_processor                 | 数据标注处理器             | callable | 否       | 自定义功能。为 None 时禁用。见数据标注处理器介绍。                                           |
| description_processor             | 数据描述处理器             | callable | 否       | 自定义功能。为 None 时禁用。见数据描述处理器介绍。                                           |
| vae_batch_size                    | VAE 批量大小               | int      | 否       |                                                                                              |
| cache_latents                     | 缓存潜变量                 | bool     | 否       | 启用时，将缓存并使用缓存的潜变量参与训练。以内存和训练前的准备换取训练速度。非常建议启用。   |
| cache_latents_to_disk             | 缓存潜变量到磁盘           | bool     | 否       | 启用时，将缓存的潜变量保存到磁盘。非常建议启用，除非您愿意承担报错而导致缓存结果丢失的后果。 |
| check_cache_validity              | 检查缓存有效性             | bool     | 否       | 启用时，将提前检查缓存文件是否有效，若您确保有效则可选择关闭以节省时间。                     |
| keep_cached_latents_in_memory     | 保持缓存潜变量在内存中     | bool     | 否       | 启用时，将加载后的潜变量保存到内存中，以内存换速度。                                         |
| async_cache                       | 异步缓存                   | bool     | 否       | 启用时，稍微加速缓存潜变量到磁盘。                                                           |

# 参数介绍

## 输出路径

训练器的输出路径由参数 `output_dir` 指定。您可以使用格式化字符串来指定输出路径。输出文件夹内包含以下子目录

- `models`：保存训练过程中的模型文件。
- `samples`：保存训练过程中的采样文件。
- `logs`：保存训练过程中的日志文件。

## VAE 精度

SDXL 官方发布的 VAE 存在缺陷，即在半精度（fp16）时会输出纯黑图像（nan）。

为了解决该问题，您可以使用参数 `no_half_vae` 禁用半精度训练，但这会导致显存占用增加。
或者，您可以使用[修复后的 VAE 模型](https://civitai.com/models/101055/sd-xl)，下载后将其路径填入 `vae` 参数中。

## 损失记录器

本训练器在训练中除了记录当前步数的损失外，还会额外记录以下两种损失：

- 平均损失：记录了最近 `stride` 步的平均损失。
- 平均损失的移动平均：记录了从训练开始至今为止的指数移动平均损失，即越近的损失值贡献越大，遗忘因子为 `gamma`。`gamma` 越大，遗忘水平越高，平均损失越接近当前损失。

## 学习率和优化器

训练的学习率和优化器高度相关。以下是几种受欢迎的搭配：

| 优化器类型 | 优化器参数                                                    | 学习率调度器         | 学习率 | 预热步数 | 学习率参数 | 说明                                               |
| ---------- | ------------------------------------------------------------- | -------------------- | ------ | -------- | ---------- | -------------------------------------------------- |
| Adafactor  | relative_step=False, scale_parameter=False, warmup_init=False | constant_with_warmup | 1e-5   | 250      |            | 适用于[等效批量](#等效批量)为 64~ 128 的大批量训练 |
| AdamW      |                                                               | cosine_with_restarts | 1e-5   | 250      |            | 适用于[等效批量](#等效批量)为 32~ 64 的小批量训练  |

## 等效批量

训练中的最终批量大小并非所设定的 `batch_size`，而是 `batch_size * gradient_accumulation_steps * num_processes`。其中，`num_processes` 为显卡数量。
一般来说，等效批量大小每提高 N 倍，学习率应当提高根号 N 倍。

## 混合精度

选择原则：尽量选择 fp16，若出现问题（如梯度爆炸、梯度消失等）再选择 bf16。一般不考虑全精度。

| 混合精度 | 说明                                                                      |
| -------- | ------------------------------------------------------------------------- |
| None     | float32 全精度训练。通常用于显存较大的情况。                              |
| fp16     | float16 混合精度训练。通常用于显存较小的情况。与 bf16 相比精度更高。      |
| bf16     | bfloat16 混合精度训练。通常用于显存较小的情况。与 fp16 相比表示范围更大。 |

## 时间步采样

传统扩散模型在训练时会随机抽取一个介于 0~1000 的整数作为时间步来训练，但在实际训练中，完全的随机抽取并不是最优策略。
在最新的研究中发现，对模型生成图像影响最大的是中段时间步，因此，提高这些时间步被抽取的概率将有助于改善训练效果。
SD3 论文中的 Rectified Flow (RF) 将时间步的采样分布从原本的均匀分布改为逻辑正态分布（logistic normal distribution），并取得了不错的效果。
虽然由于应用 RF 要求训练基模同样使用了 RF，因此 SDXL 微调并不支持完整的 RF。但是，其核心提高中段时间步权重的思想仍然值得借鉴。

本脚本实验性地实现了逻辑正态分布的时间步采样策略，您可以通过参数 `timestep_sampler_type` 和 `timestep_sampler_kwargs` 来启用和调整这一策略。具体方法为：

1. 设置 `timestep_sampler_type` 为 `logit_normal`。
2. 设置 `timestep_sampler_kwargs` 为一个字典，包含以下参数：
   - `mu`：逻辑正态分布的均值。均值越高，对高时间步的采样越多。推荐为 0 或 1.0986。
   - `sigma`：逻辑正态分布的标准差。推荐为 1。

## 重复次数获取器

该功能允许您编辑自定义函数来计算每个数据在一个 epoch 内的重复次数，以控制不同数据的占比。当设为 None 时，所有数据的重复次数均为 1。

您编辑的函数必须接受两个参数 `img_key` 和 `img_md`，另必须具有 `**kwargs` 参数以接受额外参数。返回一个整数类型的重复次数。重复次数为 0 时，该数据将被忽略。

| 参数                            | 类型  | 说明                                                                                   |
| ------------------------------- | ----- | -------------------------------------------------------------------------------------- |
| img_key                         | str   | 图像的键值，即图像文件的文件名（去除后缀）。                                           |
| img_md                          | dict  | 图像的元数据，即图像文件的元数据。                                                     |
| img_md['image_path']            | str   | 图像文件的路径。                                                                       |
| img_md['caption']               | str   | 图像的标注，即标签标注。                                                               |
| img_md['description']           | str   | 图像的描述，即自然语言标注。                                                           |
| img_md['artist']                | str   | 图像的作者。                                                                           |
| img_md['characters']            | str   | 图像的角色名单，角色之间用逗号分隔。                                                   |
| img_md['styles']                | str   | 图像的风格名单，风格之间用逗号分隔。                                                   |
| img_md['quality']               | str   | 图像的质量。                                                                           |
| img_md['aesthetic_score']       | float | 图像的美学评分。                                                                       |
| img_md['perceptual_hash']       | str   | 图像的感知哈希。                                                                       |
| img_md['safe_rating']           | float | 图像的安全评分。                                                                       |
| img_md['safe_level']            | str   | 图像的安全分级。                                                                       |
| kwargs['counter']               | dict  | 计数器，用于记录整个数据集各数据艺术家、角色、风格和质量的出现次数。                   |
| kwargs['counter']['artist']     | dict  | 记录整个数据集艺术家出现次数的计数器，键为格式化后的艺术家名，值为艺术家名的出现次数。 |
| kwargs['counter']['characters'] | dict  | 记录整个数据集角色出现次数的计数器，键为格式化后的角色名，值为角色名的出现次数。       |
| kwargs['counter']['styles']     | dict  | 记录整个数据集风格出现次数的计数器，键为格式化后的风格名，值为风格名的出现次数。       |
| kwargs['counter']['quality']    | dict  | 记录整个数据集质量出现次数的计数器，键为质量名，值为质量名的出现次数。                 |

其中，“格式化” 意为将一个标签的空格替换为下划线，并还原用斜杠转义的括号。

## 数据标注处理器和数据描述处理器

该功能允许您编辑自定义函数来在每个数据每次被取出时对其标注和描述进行的额外处理方式。当设为 None 时，所有数据的标注和描述均不做额外处理。

您编辑的函数必须接受一个参数 `img_info`，另必须具有 `**kwargs` 参数以接受额外参数。返回一个字符串类型的图像标注。

| 参数                   | 类型            | 说明                                                   |
| ---------------------- | --------------- | ------------------------------------------------------ |
| img_info               | ImageInfo       | 图像数据的所有信息。                                   |
| img_info.key           | str             | 图像的键值，即图像文件的文件名（去除后缀）。           |
| img_info.metadata      | dict            | 图像的元数据，同[上表](#重复次数获取器)中的 `img_md`。 |
| img_info.caption       | str             | 图像的标注。                                           |
| img_info.description   | str             | 图像的描述。                                           |
| img_info.image_size    | tuple[int, int] | 图像的尺寸。                                           |
| img_info.original_size | tuple[int, int] | 图像的原始尺寸。                                       |
| img_info.latent_size   | tuple[int, int] | 图像的潜变量尺寸。                                     |
| img_info.bucket_size   | tuple[int, int] | 图像的分桶尺寸。                                       |
| img_info.num_repeats   | int             | 图像的重复次数。                                       |
| img_info.npz_path      | str             | 图像的 npz 文件路径。                                  |
| kwargs['counter']      | dict            | 计数器，同[上表](#重复次数获取器)。                    |
| kwargs['flip_aug']     | bool            | 该图像在这次加载时是否经过了随机水平翻转。             |
