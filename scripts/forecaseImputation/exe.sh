export CUDA_VISIBLE_DEVICES=0

# model_name=Transformer_AR
model_name=TimesNet_AR
# model_name=iTransformer_AR
rate=0.3
seeds=(1)
miss_lens=(100)
# miss_lens=(100 300)


##############################################
for miss_len in "${miss_lens[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/guangzhou/ \
      --data_path guangzhou.csv \
      --model_id guangzhou \
      --model $model_name \
      --data 3MAutoData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 96 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 214 \
      --dec_in 214 \
      --c_out 214 \
      --batch_size 16 \
      --d_model 256 \
      --d_ff 256 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed \
      --miss_len $miss_len \
      --max_miss_len 10000000
  done
done
