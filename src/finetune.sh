#!/bin/bash
# For runpod
#root_dir = /workspace/futs
# For local
root_dir=/Users/tonywy/Desktop/Xode/mvts

python main.py --output_dir $root_dir/output/futs/finetune/$1 --name futs_finetune_$1 --records_file Regression_records.xls --data_class futs --epochs 100 --lr 0.001 --optimizer RAdam  --pos_encoding learnable --task regression --data_dir $root_dir/data/ --pattern "ZCE_CH_$1/train/daily_frame.*.parquet" --val_pattern "ZCE_CH_$1/val/daily_frame.*.parquet" --load_model $root_dir/output/futs/pretrain/futs_imputation_2024-01-20_16-53-34_KEd/checkpoints/model_best.pth --change_output
# stop the pod when the job is finshed
#runpodctl stop pod $RUNPOD_POD_ID