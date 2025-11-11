seq_len=168
model_name=DLinear

root_path=./dataset/mydata_v1/
data_path=h57.csv
model_id_name=h57
data_name=custom
des=Exp-D
for pred_len in 24 48 96 168
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
      --enc_in 57\
      --dropout 0.1\
      --train_epochs 20\
      --patience 10\
      --des $des \
      --itr 1 \
      --learning_rate 0.01 \
      --ps_lambda 10.0 \
      --use_ps_loss 0
done