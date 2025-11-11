#export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer
ps_lambdas=(2.0 6.0 10.0)
seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
batch_size=16
train_epochs=20
patience=10
f=57
for pred_len in 24 48 96 168
do
  for ps_lambda in ${ps_lambdas[@]}
  do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/mydata_v1/ \
      --data_path h57.csv \
      --model_id h57 \
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
      --des pw$ps_lambda \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size 16 \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window \
      --ps_lambda $ps_lambda \
      --use_ps_loss 1
  done
done

