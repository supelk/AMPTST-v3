export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/othermodel" ]; then
    mkdir ./logs/othermodel
fi
f=57
model_name=SegRNN

seq_len=96
for pred_len in 24 168
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/mydata/ \
    --data_path h57.csv \
    --model_id h57 \
    --model $model_name \
    --data custom \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --seg_len 48 \
    --enc_in $f \
    --d_model 512 \
    --dropout 0.5 \
    --learning_rate 0.0001 \
    --des 'Exp' \
    --itr 1 >logs/othermodel/$model_name'_'$seq_len'_'$pred_len'_h57'.log
done