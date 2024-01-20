#!bin/bash

python src/main.py --output_dir ../output/futs/ --name futs_imputation --records_file Regression_records.xls --data_class futs --epochs 100 --lr 0.001 --optimizer RAdam  --pos_encoding learnable --task imputation --data_dir "/Users/tonywy/Desktop/Xode/futs_data/ur/daily_frame.*.parquet"