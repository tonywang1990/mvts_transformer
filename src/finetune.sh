#!/bin/bash
# For runpod
root_dir=/workspace/futs
# For local
#root_dir=/Users/tonywy/Desktop/Xode/mvts

## Memory calcualtion
# 1 batch of data = batch_size * seq_len * feature_dim 
# (256 * 1024 * 24) * 4 bytes = 24Gb
# model = foward + backward
# (282625 + 282625) * 4 bytes = 2Gb
# total:
# 26 Gb

#python main.py --output_dir $root_dir/output/futs/finetune/$1 --name futs_finetune_$1 --records_file Regression_records.xls --data_class futs --epochs 10 --lr 0.001 --optimizer RAdam  --pos_encoding learnable --task regression --data_dir $root_dir/data/ --pattern "ZCE_CH_$1/train/daily_frame.*.parquet" --val_pattern "ZCE_CH_$1/val/daily_frame.*.parquet" --load_model $root_dir/output/futs/pretrain/futs_imputation_2024-01-20_16-53-34_KEd/checkpoints/model_best.pth --change_output

python main.py --output_dir $root_dir/output/mvts/finetune/$1 --name futs_finetune_$1 --records_file Regression_records.xls --data_class futs --epochs 10 --lr 0.0001 --batch_size 256 --optimizer RAdam  --pos_encoding learnable --task regression --data_dir $root_dir/data/ --pattern "ZCE_CH_$1/train/daily_frame.*.parquet" --val_pattern ZCE_CH_$1/val/daily_frame.202312??.parquet --sampling_ratio 0.3 --load_model $root_dir/output/mvts/pretrain/futs_imputation_2024-01-23_20-50-12_foundation/checkpoints/model_best.pth --change_output


# For testing only - overfitting one day
#python main.py --output_dir $root_dir/output/mvts/finetune/UR --name futs_finetune_UR --records_file Regression_records.xls --data_class futs --epochs 10 --lr 0.001 --batch_size 64 --optimizer RAdam  --pos_encoding learnable --task regression --data_dir $root_dir/data/ --pattern "ZCE_CH_UR/train/daily_frame.20230104.parquet" --val_pattern "ZCE_CH_UR/train/daily_frame.20230104.parquet" --load_model $root_dir/output/mvts/finetune/UR/futs_finetune_UR_2024-01-27_19-47-48_w6i/checkpoints/model_best.pth --sampling_ratio 0.1


echo "Idling... Counting down 60 seconds before stopping the pod:"
read -t 60 userInput

if [ -n "$userInput" ]; then
    echo "Exiting stopped"
else
    echo "Stopping pod..."
    # stop the pod when the job is finshed
    runpodctl stop pod $RUNPOD_POD_ID
fi
