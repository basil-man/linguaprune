NUM_GPUS=1  # Total number of GPUs available
BATCH_SIZE=4  # Adjust batch size if needed
GPU_OFFSET=0  # Starting GPU index

MODEL_NAME=./checkpoints-merged/Qwen2.5-0.5B-Instruct-300/global_step_280
DATASET=gsm8k

BUDGETS=("300")
for BUDGET in ${BUDGETS[@]}
do
    SAVE_NAME="qwen2.5_0.5B-budget${BUDGET}"
    for RANK in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$((GPU_OFFSET + RANK)) python tools/generation_tools/budget_forcing_gen.py \
            --rank=$RANK \
            --world_size=$NUM_GPUS \
            --batch_size=$BATCH_SIZE \
            --dataset_name=$DATASET \
            --max_tokens_thinking=$BUDGET \
            --orig_model_name=Qwen/Qwen2.5-0.5B-Instruct \
            --save_name=$SAVE_NAME \
            --model_name=$MODEL_NAME &
    done
    wait
done