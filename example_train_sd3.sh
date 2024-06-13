accelerate launch train_sd3.py --config=configs/condif_train_sd3.py # 单卡训练
accelerate launch --num_processes=4 --multi_gpu --gpu_ids=0,1,2,3 train_sd3.py --config=configs/condif_train_sd3.py # 单机多卡训练