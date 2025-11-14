export CUDA_VISIBLE_DEVICES=0
f=57
model_name=PatchTST
seq_len=168
for pred_len in 24 48 96 168 192
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
    --learning_rate 0.01 \
    --des 'psl' \
    --itr 1 \
    --n_heads 4 \
    --use_ps_loss 1 \
    --head_or_projection 0
done