export CUDA_VISIBLE_DEVICES=2

model_name=Transformer_AR
# model_name=TimesNet_AR
rates=(0.1)
seeds=(2 3 4)

# ##############################################
# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_GPT \
#       --is_training 1 \
#       --root_path ./dataset/traffic/ \
#       --data_path traffic.csv \
#       --model_id traffic \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 2 \
#       --factor 3 \
#       --enc_in 862 \
#       --dec_in 862 \
#       --c_out 862 \
#       --batch_size 16 \
#       --d_model 512 \
#       --d_ff 512 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --AR_len 1 \
#       --GPT_len 48
#   done
# done

# ##############################################
# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_GPT \
#       --is_training 1 \
#       --root_path ./dataset/metr-la/ \
#       --data_path metr-la.csv \
#       --model_id metr-la \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 2 \
#       --factor 3 \
#       --enc_in 207 \
#       --dec_in 207 \
#       --c_out 207 \
#       --batch_size 16 \
#       --d_model 128 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --AR_len 1 \
#       --GPT_len 48
#   done
# done

##############################################
for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name forecastImputation_GPT \
      --is_training 1 \
      --root_path ./dataset/TCPC/ \
      --data_path TCPC.csv \
      --model_id TCPC \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 144 \
      --e_layers 2 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 16 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed \
      --AR_len 1 \
      --GPT_len 48
  done
done