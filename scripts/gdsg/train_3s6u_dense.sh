export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u msco_train.py \
  --task "msco" \
  --project_name "GDSG" \
  --wandb_logger_name "msco_dense_hybrid_3s6u" \
  --diffusion_type "hybrid" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/root/GDSG/data/msmu-co" \
  --training_split "3server/3s6u_60000samples_20240606212245.txt" \
  --validation_split "3server/3s6u_2000samples_20240606182344.txt" \
  --test_split "3server/3s6u_2000samples_20240606182344.txt" \
  --batch_size 256 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --diffusion_steps 200 \
  --inference_diffusion_steps 5 \
  --hidden_dim 256 \
  --n_layers 5