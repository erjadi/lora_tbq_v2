#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="../../cog_instance_data"
export OUTPUT_DIR="../../cog_output"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$5  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt=$1 \
  --resolution=$3 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=$4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$2 \
  --mixed_precision=fp16