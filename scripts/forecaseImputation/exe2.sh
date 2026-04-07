export CUDA_VISIBLE_DEVICES=5

# model_name=Transformer_AR
model_name=TimesNet_AR
# model_name=iTransformer_AR
rate=0.3
seeds=(1 2 3)
miss_lens=(30 50 100 300 500)


# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
#       --is_training 1 \
#       --root_path ./dataset/Exchange/ \
#       --data_path exchange_rate.csv \
#       --model_id exchange_rate \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 8 \
#       --dec_in 8 \
#       --c_out 8 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 16 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done

# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
#       --is_training 1 \
#       --root_path ./dataset/weather/ \
#       --data_path weather.csv \
#       --model_id Weather \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 21 \
#       --dec_in 21 \
#       --c_out 21 \
#       --batch_size 16 \
#       --d_model 64 \
#       --d_ff 64 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done

##############################################
for miss_len in "${miss_lens[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name forecastImputation_3M \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id electricity \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 144 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --batch_size 16 \
      --d_model 128 \
      --d_ff 128 \
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



# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh1.csv \
#       --model_id ETTh1 \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done

# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh2.csv \
#       --model_id ETTh2 \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done

# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm1.csv \
#       --model_id ETTm1 \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done

# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm2.csv \
#       --model_id ETTm2 \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done

# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
#       --is_training 1 \
#       --root_path ./dataset/TCPC/ \
#       --data_path TCPC.csv \
#       --model_id TCPC \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 8 \
#       --dec_in 8 \
#       --c_out 8 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 16 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done

# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
#       --is_training 1 \
#       --root_path ./dataset/pm25/ \
#       --data_path pm25.csv \
#       --model_id pm25 \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 144 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 36 \
#       --dec_in 36 \
#       --c_out 36 \
#       --batch_size 16 \
#       --d_model 32 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done

# ##############################################
# for miss_len in "${miss_lens[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M \
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
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 207 \
#       --dec_in 207 \
#       --c_out 207 \
#       --batch_size 16 \
#       --d_model 256 \
#       --d_ff 256 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
#       --miss_len $miss_len \
#       --max_miss_len 10000000
#   done
# done
