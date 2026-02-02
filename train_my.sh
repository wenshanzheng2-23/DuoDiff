    # --report_to "wandb" \
    #     --max_train_steps 50000 \
    #    --logging_dir ./logs \
# export WANDB_HTTP_PROXY=http://127.0.0.1:7890
# export WANDB_HTTPS_PROXY=http://127.0.0.1:7890
# wandb sync ./outputs/difix/unet_den/wandb/offline-run-20250917_170452-m12pug2s
# export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
python  amyself/train.py \
    --output_dir=./outputs/aaa/real10_test \
    --dataset_path="data/realestate10k_6_16/10.json" \
    --num_training_epochs 502 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --gradient_accumulation_steps 8\
    --dataloader_num_workers 16 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing\
    --checkpointing_steps=1000 \
    --eval_freq 200 \
    --viz_freq 100 \
    --num_samples_eval 10 \
    --lambda_lpips 1.5 \
    --lambda_l2 0.5 \
    --lambda_gram 1.0 \
    --gram_loss_warmup_steps 5002 \
    --tracker_project_name "dec_unet" \
    --tracker_run_name "10_test" \
    --report_to "wandb" \
    --timestep 199 \
    --mixed_precision=bf16 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=200 \
    # --resume="outputs/aaa/twenty_gram/checkpoints/model_12501.pkl" \

    
    # ZeroConv