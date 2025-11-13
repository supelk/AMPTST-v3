seq_len=96
model_name=DLinear

root_path=./dataset/mydata_v1/
data_path=h57summer.csv
model_id_name=h57s
data_name=custom
pred_len = 96

for lreaning_rate in 0.01 0.001 0.0001 0.00001
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --root_path $root_path \
      --data_path $data_path \
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 57\
      --dropout 0.1\
      --des lr-$learning_rate \
      --itr 1 \
      --learning_rate $learning_rate
done