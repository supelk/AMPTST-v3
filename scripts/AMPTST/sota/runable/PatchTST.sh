export CUDA_VISIBLE_DEVICES=0
f=57
model_name=PatchTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata_v1/ \
  --data_path h57.csv \
  --model_id h57_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --patience 10 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --des 'ExpPS9' \
  --itr 1 \
  --n_heads 4 \
  --ps_lambda 9.0 \
  --use_ps_loss 1 \
  --head_or_projection 0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata_v1/ \
  --data_path h57.csv \
  --model_id h57_168_168 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 168 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --patience 10 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --des 'ExpPS9' \
  --itr 1 \
  --n_heads 4 \
  --ps_lambda 9.0 \
  --use_ps_loss 1 \
  --head_or_projection 0