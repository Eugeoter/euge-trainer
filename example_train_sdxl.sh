bash prepare.sh
accelerate launch train_sdxl.py --config=configs/condif_train_sdxl.py # 单卡训练
accelerate launch --num_processes=4 --multi_gpu --gpu_ids=0,1,2,3 train_sdxl.py --config=configs/condif_train_sdxl.py # 单机多卡训练