export CUDA_VISIBLE_DEVICES=0
model_name=AMPTST-v4

seq_len=96
e_layers=3

d_model=64
d_ff=128

f=321
data_path=electricity.csv
des=Exp
for pred_len in 96 192
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id electricity \
    --model $model_name \
    --data custom \
    --root_path ./dataset/electricity/ \
    --data_path $data_path \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --top_k 3 \
    --enc_in $f \
    --dec_in $f \
    --c_out $f \
    --d_model $d_model \
    --n_heads 8 \
    --e_layers $e_layers \
    --d_layers 1 \
    --d_ff $d_ff \
    --factor 3 \
    --channel_independence 0 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --des $des \
    --itr 1 \
    --pf 0 
done