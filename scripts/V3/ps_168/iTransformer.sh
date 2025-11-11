export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/othermodel" ]; then
    mkdir ./logs/othermodel
fi
f=33
seq_len=168
model_name=iTransformer
ps_lambdas=(2.0 6.0 10.0)
for pred_len in 24 48 96 168
do
  for ps_lambda in ${ps_lambdas[@]}
  do
    python  -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/mydata_v1/ \
      --data_path h33.csv \
      --model_id h33 \
      --model $model_name \
      --data custom \
      --features MS \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in $f \
      --dec_in $f \
      --c_out $f \
      --des Exp$ps_lambda \
      --d_model 32 \
      --d_ff 32 \
      --itr 1 \
      --ps_lambda $ps_lambda \
      --use_ps_loss 1 \
      --head_or_projection 1
    done
done

