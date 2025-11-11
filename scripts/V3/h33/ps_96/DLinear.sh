seq_len=96
model_name=DLinear
ps_lambda=3.0
root_path=./dataset/mydata_v1/
data_path=h33.csv
model_id_name=h33
data_name=custom

#for pred_len in 24 48 96 168
for pred_len in 192
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 33\
      --dropout 0.1\
      --des Exp$ps_lambda \
      --itr 1 \
      --ps_lambda $ps_lambda \
      --use_ps_loss 1
done