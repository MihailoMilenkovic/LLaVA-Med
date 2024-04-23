script_dir=$(dirname "$(readlink -f "$0")")
par_dir=$(dirname "$script_dir")
weight_dir_og=$HOME/llava-weights-og
weight_dir_finetuned=$HOME/llava-weights-finetuned-vqa-rad
repo_dir=$HOME/git/LLaVA-Med

epochs=5
lr=2e-5

if [ "$1" == "--epochs" ]; then
    epochs="$2"
    shift 2
fi

if [ "$1" == "--lr" ]; then
    lr="$2"
    shift 2
fi

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
    $par_dir/llava/train/train_mem.py \
    --model_name_or_path $weight_dir_og \
    --data_path $repo_dir/data/vqa_rad/train/data.json \
    --image_folder $repo_dir/data/vqa_rad/train/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir $weight_dir_finetuned \
    --num_train_epochs $epochs \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb    

    # --pretrain_mm_mlp_adapter /home/chunyl/research/models/llava/LLaVA-13b-pretrain-projector-v0/LLaVA-13b-pretrain-projector-v0-CC3M-595K-original_caption.bin \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \