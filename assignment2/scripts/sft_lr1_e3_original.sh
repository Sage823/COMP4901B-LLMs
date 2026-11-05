export WANDB_API_KEY=""

export WANDB_PROJECT="COMP4901B-Homework2"
RUNNAME="HW2_lr1_e3_original"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="135M"
OUTPUTPATH="ckpt"
DEVICES="0"
NUM_GPUS=1

# Match the skeleton batch/length settings so this experiment differs from the
# baseline mainly in learning rate and the explicit three-epoch schedule.
TOTALBSZ=128
BSZPERDEV=1
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))
export CUDA_VISIBLE_DEVICES=${DEVICES}
echo "Training model ${MODEL_SIZE} using ${NUM_GPUS} GPUs, ${BSZPERDEV} batch size per GPU, ${GRADACC} gradient accumulation steps"

python train_hw_parallel.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --report_to "none" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --fp16 False \
    --flash_attn False \
    --dataloader_num_workers 2 \
    --preprocess_workers 2 \
    --max_rounds 5
