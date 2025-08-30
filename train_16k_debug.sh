export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export MASTER_PORT=$((10000 + RANDOM % 20000))
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 accelerate launch --mixed_precision=bf16 --multi_gpu \
  --num_machines 1 --num_processes 8 src/train_difix.py \
  --output_dir=./outputs/difix/train_small_bs2_8gpu_640_2views_16k_debug \
  --dataset_path="/data2/peilincai/Difix3D/datasets_16k/difix3d.json" \
  --max_train_steps 24000 \
  --num_training_epochs 10 \
  --resolution=640 --learning_rate 1e-5 \
  --train_batch_size=1 --dataloader_num_workers 8 \
  --gradient_checkpointing \
  --checkpointing_steps=1800 --eval_freq=100 --viz_freq=50 \
  --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 0.25 --gram_loss_warmup_steps 2000 \
  --lpips_net vgg \
  --views_per_microbatch 2 \
  --use_8bit_optimizer \
  --freeze_unet \
  --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train_small_640_2views_16k_debug" \
  --timestep 199 \
  --resume /data2/peilincai/Difix3D/outputs/difix/train_small_bs2_8gpu_640_2views_16k_debug/checkpoints/model_5400.pkl --lr_warmup_steps 0