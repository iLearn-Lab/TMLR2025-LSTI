export CUDA_VISIBLE_DEVICES=5

# model_name=Transformer_AR
model_name=TimesNet_AR
# model_name=iTransformer_AR
rates=(0.3)
# rate=0.1
seeds=(2)
# miss_lens=(10 30 50 100 300)
miss_len=50
# types=(Disjoint Overlap MCAR_B Blackout)
types=(Disjoint Overlap MCAR_B)
# types2=(Disjoint Overlap MCAR_B)
type=Blackout


# ##############################################
# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M_auto_plus \
#       --is_training 1 \
#       --root_path ./dataset/guangzhou/ \
#       --data_path guangzhou.csv \
#       --model_id guangzhou \
#       --model $model_name \
#       --data 3MAutoData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 96 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 214 \
#       --dec_in 214 \
#       --c_out 214 \
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
#       --max_miss_len 10000000 \
#       --missing_type $type
#   done
# done


##############################################
for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name forecastImputation_3M_auto_plus \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id electricity \
      --model $model_name \
      --data 3MAutoData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 96 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
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
      --max_miss_len 10000000 \
      --missing_type $type
  done
done

# ##############################################
# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M_auto_plus \
#       --is_training 1 \
#       --root_path ./dataset/PEMS/ \
#       --data_path PEMS04.csv \
#       --model_id PEMS04 \
#       --model $model_name \
#       --data 3MAutoData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 96 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 307 \
#       --dec_in 307 \
#       --c_out 307 \
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
#       --max_miss_len 10000000 \
#       --missing_type $type
#   done
# done

# ##############################################
# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M_auto_plus \
#       --is_training 1 \
#       --root_path ./dataset/metr-la/ \
#       --data_path metr-la.csv \
#       --model_id metr-la \
#       --model $model_name \
#       --data 3MAutoData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 96 \
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
#       --max_miss_len 10000000 \
#       --missing_type $type
#   done
# done

# ##############################################
# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name forecastImputation_3M_auto_plus \
#       --is_training 1 \
#       --root_path ./dataset/traffic/ \
#       --data_path traffic.csv \
#       --model_id traffic \
#       --model $model_name \
#       --data 3MAutoData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 96 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 862 \
#       --dec_in 862 \
#       --c_out 862 \
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
#       --max_miss_len 10000000 \
#       --missing_type $type
#   done
# done


