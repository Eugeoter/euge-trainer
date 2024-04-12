accelerate launch cache_latents.py --config=configs/train_config.py # 单卡缓存
accelerate launch --num_processes=4 --multi_gpu --gpu_ids=0,1,2,3 cache_latents.py --config=configs/train_config.py # 单机多卡缓存