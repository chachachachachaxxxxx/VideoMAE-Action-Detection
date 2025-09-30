# Set the path to save checkpoints and logs (single-GPU eval)
OUTPUT_DIR='/storage/wangxinxing/code/VideoMAE-Action-Detection/runs/ava_videomae_vit_small_k400_pretrain+finetune_single_eval'
# path to pretrain/finetuned model
# Google Drive Link: https://drive.google.com/file/d/1ygjLRm1kvs9mwGsP3lLxUExhRo6TWnrx
MODEL_PATH='/storage/wangxinxing/model/videomae_vit_small_k400_pretrain+finetune/checkpoint.pth'

# Optional: restrict to one GPU explicitly
# export CUDA_VISIBLE_DEVICES=0

OMP_NUM_THREADS=1 python3 run_class_finetuning.py \
      --model vit_small_patch16_224 \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 8 \
      --input_size 224 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 5e-4 \
      --layer_decay 0.6 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --data_set "ava" \
      --drop_path 0.2 \
      --val_freq 10 \
      --eval

