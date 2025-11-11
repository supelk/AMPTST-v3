export CUDA_VISIBLE_DEVICES=0
model_name=TimesNet
f=57
seq_len=96
pred_len=96
for batch_size in 32 64
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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size $batch_size \
  --itr 1 \
  --ps_lambda 10.0 \
  --use_ps_loss 0 \
  --head_or_projection 1
done

