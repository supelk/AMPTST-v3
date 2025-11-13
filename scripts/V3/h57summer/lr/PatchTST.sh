export CUDA_VISIBLE_DEVICES=0
f=57
model_name=PatchTST
seq_len=96
pred_len=96
for lreaning_rate in 0.01 0.001 0.0001 0.00001
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/mydata_v1/ \
    --data_path h57summer.csv \
    --model_id h57s \
    --model $model_name \
    --data custom \
    --features MS \
    --seq_len $seq_len \
    --label_len 48 \
    --d_model 32 \
    --d_ff 32 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patience 10 \
    --enc_in $f \
    --dec_in $f \
    --c_out $f \
    --learning_rate $learning_rate \
    --des lr-$learning_rate \
    --itr 1 \
    --n_heads 4
done