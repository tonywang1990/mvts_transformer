#!/bin/bash
# For runpod
python main.py --output_dir ../output/futs/ --name futs_imputation --records_file Regression_records.xls --data_class futs --epochs 100 --lr 0.001 --optimizer RAdam  --pos_encoding learnable --task imputation --data_dir "/workspace/futs/data/" --pattern "ZCE_CH_*/train/daily_frame.*.parquet" --val_pattern "ZCE_CH_*/val/daily_frame.*.parquet"
# stop the pod when the job is finshed
#runpodctl stop pod $RUNPOD_POD_ID