accelerate launch sdxl_train.py --config=configs/train_config.py # 单卡训练
accelerate launch --num_processes=4 --multi_gpu --gpu_ids=0,1,2,3 sdxl_train.py --config=configs/train_config.py # 单机多卡训练