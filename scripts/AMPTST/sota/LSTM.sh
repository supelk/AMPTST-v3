# add --individual for DLinear-I
seq_len=168
model_name=TSTDLSTM

root_path=./dataset/mydata_v1/
data_path=h57.csv
model_id_name=h57
data_name=custom
f=57
for pred_len in 24 168
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 0 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in $f\
      --dropout 0.2\
      --patience 10 \
      --des exp \
      --itr 1 --batch_size 32 --learning_rate 0.001
done