export CUDA_VISIBLE_DEVICES=0
model_name=TimesNet
ps_lambda=3.0
f=33
seq_len=96
#for pred_len in 24 48 96 168
for pred_len in 192
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/mydata_v1/ \
      --data_path h33.csv \
      --model_id h33 \
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
      --top_k 5 \
      --des Exp$ps_lambda \
      --itr 1 \
      --ps_lambda $ps_lambda \
      --use_ps_loss 1 \
      --head_or_projection 1
done

