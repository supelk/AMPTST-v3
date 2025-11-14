export CUDA_VISIBLE_DEVICES=0
f=57
seq_len=168
model_name=iTransformer
for pred_len in 24 48 96 168 192
do
python  -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id h57s1\
  --model $model_name \
  --data custom \
  --root_path ./dataset/mydata_v1/ \
  --data_path h57summer1.csv \
  --features MS \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --des 'psl' \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --itr 1 \
  --use_ps_loss 1 \
  --head_or_projection 1
done

