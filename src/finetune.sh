#!/bin/bash
# For runpod
root_dir=/workspace/futs
# For local
#root_dir=/Users/tonywy/Desktop/Xode/mvts

python main.py --output_dir $root_dir/output/futs/finetune/$1 --name futs_finetune_$1 --records_file Regression_records.xls --data_class futs --epochs 20 --lr 5e-4 --lr_step 5 --optimizer RAdam  --pos_encoding learnable --task regression --data_dir $root_dir/data/ --pattern "ZCE_CH_$1/train/daily_frame.*.parquet" --val_pattern "ZCE_CH_$1/val/daily_frame.*.parquet" --load_model $root_dir/output/futs/pretrain/futs_imputation_2024-01-23_20-50-12_foundation/checkpoints/model_best.pth --change_output

# Resume from prev checkpoint
#python main.py --output_dir $root_dir/output/futs/finetune/$1 --name futs_finetune_$1 --records_file Regression_records.xls --data_class futs --epochs 100 --lr 1e-4 --lr_step 10 --optimizer RAdam  --pos_encoding learnable --task regression --data_dir $root_dir/data/ --pattern "ZCE_CH_$1/train/daily_frame.*.parquet" --val_pattern "ZCE_CH_$1/val/daily_frame.*.parquet" --load_model $root_dir/output/futs/finetune/UR/futs_finetune_UR_2024-01-25_21-57-36_m7a/checkpoints/model_best.pth 
# stop the pod when the job is finshed
#runpodctl stop pod $RUNPOD_POD_ID

# Train from scratch - for DEBUGGING purpose
python main.py --output_dir $root_dir/output/futs/finetune/$1 --name futs_finetune_$1 --records_file Regression_records.xls --data_class futs --epochs 50 --lr 1e-3 --lr_step 5 --optimizer RAdam  --pos_encoding learnable --task regression --data_dir $root_dir/data/ --pattern "ZCE_CH_$1/train/daily_frame.*.parquet" --val_pattern "ZCE_CH_$1/val/daily_frame.*.parquet" 