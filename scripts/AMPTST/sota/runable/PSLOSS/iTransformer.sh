export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/othermodel" ]; then
    mkdir ./logs/othermodel
fi
f=57
model_name=iTransformer
for pred_len in 24 48 96 168
do
python  -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mydata/ \
  --data_path h57.csv \
  --model_id h57_168_$pred_len \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $f \
  --dec_in $f \
  --c_out $f \
  --des 'Exp_PS10_withsamehyperparameter' \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --itr 1 \
  --ps_lambda 10.0 \
  --use_ps_loss 1 \
  --head_or_projection 1
done

