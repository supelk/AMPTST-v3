export CUDA_VISIBLE_DEVICES=0
model_name=TimesNet
f=57
seq_len=96
pred_len=96
for learning_rate in 0.01 0.001 0.0001 0.00001
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata_v1/ \
  --data_path h57summer1.csv \
  --model_id h57s1\
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate $learning_rate \
  --top_k 5 \
  --des lr-$learning_rate \
  --itr 1
done

