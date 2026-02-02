    # --report_to "wandb" \
    #     --max_train_steps 50000 \
    #    --logging_dir ./logs \
# export WANDB_HTTP_PROXY=http://127.0.0.1:7890
# export WANDB_HTTPS_PROXY=http://127.0.0.1:7890
# wandb sync ./outputs/difix/unet_den/wandb/offline-run-20250917_170452-m12pug2s
# export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
accelerate launch  src/train_difix.py \
    --output_dir=./outputs/difix/unet_den4 \
    --dataset_path="data/denDL3DV9.json" \
    --num_training_epochs 500 \
    --resolution=256 \
    --learning_rate 5e-6 \
    --train_batch_size=1 \
    --gradient_accumulation_steps 8\
    --dataloader_num_workers 4 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing\
    --checkpointing_steps=5000 \
    --eval_freq 1000 \
    --viz_freq 100 \
    --lambda_lpips 1.0 \
    --lambda_l2 1.0 \
    --lambda_gram 1.0 \
    --gram_loss_warmup_steps 5000 \
    --tracker_project_name "difix_unet" \
    --tracker_run_name "mv_net" \
    --report_to "wandb" \
    --timestep 199 \
    --mixed_precision=bf16 \
    --mv_unet \
    

# 9_22 unet+vae_skip
# 9_23 unet+no_skip