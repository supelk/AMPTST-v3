export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/AMPTST/main" ]; then
    mkdir ./logs/AMPTST/main
fi
model_name=m1

seq_len=96
e_layers=3
learning_rate=0.01
d_model=32
f=57
data_path=h57.csv
des=MLP

for pred_len in 24 48 96 168
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/mydata_v1/ \
    --data_path $data_path \
    --model_id h57 \
    --model $model_name \
    --data custom \
    --features MS \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --enc_in $f \
    --des $des \
    --itr 1 \
    --d_model $d_model \
    --m1_type 1
done