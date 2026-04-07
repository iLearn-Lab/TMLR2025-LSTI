export CUDA_VISIBLE_DEVICES=3

# model_name=Transformer_AR
model_name=TimesNet_AR
# model_name=iTransformer_AR
rate=0.3
seeds=(2 3)
miss_lens=(300)


##############################################
for miss_len in "${miss_lens[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name forecastImputation_3M \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS03.csv \
      --model_id PEMS \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 144 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 358 \
      --dec_in 358 \
      --c_out 358 \
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