#!/bin/bash
# For runpod
root_dir=/workspace/futs
# For local
#root_dir=/Users/tonywy/Desktop/Xode/mvts
symbol=UR
# Resume from prev checkpoint
# Only one day of data is recommended
python main.py --output_dir $root_dir/output/futs/finetune/$symbol --name futs_finetune_$symbol --records_file Regression_records.xls --data_class futs_test --epochs 100 --lr 1e-4 --lr_step 10 --optimizer RAdam  --pos_encoding learnable --task regression --data_dir $root_dir/data/ --test_pattern ZCE_CH_$symbol/val/daily_frame.20231208.parquet --pattern ZCE_CH_$symbol/val/daily_frame.20231208.parquet --val_pattern ZCE_CH_$symbol/val/daily_frame.20231208.parquet   --load_model $root_dir/output/futs/finetune/UR/futs_finetune_UR_2024-01-25_22-59-22_TFY/checkpoints/model_best.pth --test_only testset
# stop the pod when the job is finshed
#runpodctl stop pod $RUNPOD_POD_ID
