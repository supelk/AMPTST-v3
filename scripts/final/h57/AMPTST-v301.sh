export CUDA_VISIBLE_DEVICES=0
model_name=AMPTST-v301

seq_len=96
e_layers=3

d_model=32
d_ff=32

f=57
data_path=h57.csv
des=psl
#for pred_len in 24 96 192
for pred_len in 48
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id h57 \
    --model $model_name \
    --data custom \
    --root_path ./dataset/mydata_v1/ \
    --data_path $data_path \
    --features MS \
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
    --pf 0 \
    --use_ps_loss 1 \
    --head_or_projection 1
done