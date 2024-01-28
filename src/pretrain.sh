#!/bin/bash
# For runpod
# For runpod
root_dir=/workspace/futs
# For local
#root_dir=/Users/tonywy/Desktop/Xode/mvts

python main.py --output_dir $root_dir/output/mvts/pretrain --name futs_imputation --records_file Regression_records.xls --data_class futs --epochs 100 --lr 0.001 --batch_size 256 --optimizer RAdam  --pos_encoding learnable --task imputation --data_dir $root_dir/data/ --pattern "ZCE_CH_*/train/daily_frame.*.parquet" --val_pattern "ZCE_CH_*/val/daily_frame.202312??.parquet" --sampling_ratio 0.01

##################
## Money Saver! ##
##################
echo "Idling... Counting down 60 seconds before stopping the pod:"
read -t 60 userInput

if [ -n "$userInput" ]; then
    echo "Exiting stopped"
else
    echo "Stopping pod..."
    # stop the pod when the job is finshed
    runpodctl stop pod $RUNPOD_POD_ID
fi
