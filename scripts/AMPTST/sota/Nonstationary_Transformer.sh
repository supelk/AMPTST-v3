export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/othermodel
fi
f=57
model_name=Nonstationary_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata/ \
  --data_path h57.csv \
  --model_id h57 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 >logs/othermodel/$model_name'_168_24_h57'.log

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata/ \
  --data_path h57.csv \
  --model_id h57 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 168 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 >logs/othermodel/$model_name'_168_168_h57'.log