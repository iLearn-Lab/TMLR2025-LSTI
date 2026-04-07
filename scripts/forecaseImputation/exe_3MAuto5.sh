export CUDA_VISIBLE_DEVICES=3

# model_name=Transformer_AR
model_name=TimesNet_AR
# model_name=iTransformer_AR
rates=(0.1 0.3 0.5 0.7)
# rate=0.1
seeds=(1)
# miss_lens=(10 30 50 100 300)
miss_len=50
# types=(Disjoint Overlap MCAR_B Blackout)
types=(Disjoint Overlap MCAR_B)
# types2=(Disjoint Overlap MCAR_B)
type=Blackout


##############################################
for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name forecastImputation_3M_auto_plus \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id ETTm1 \
      --model $model_name \
      --data 3MAutoData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 96 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
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


