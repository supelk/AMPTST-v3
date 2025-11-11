export CUDA_VISIBLE_DEVICES=0
model_name=TimeMixer

seq_len=168
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=20
patience=10
f=57
data_path=h57.csv
model_id=h57
des=ci0
for pred_len in 24 168
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/mydata/ \
    --data_path $data_path \
    --model_id $model_id \
    --model $model_name \
    --data custom \
    --features MS \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --enc_in $f \
    --dec_in $f \
    --c_out $f \
    --des $des \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --moving_avg 25 \
    --batch_size 32 \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window >logs/myforecasting/$model_name'_'$model_id'_'$seq_len'_'$pred_len'_'$data_path'_'$des.log
done