export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u msco_train.py \
  --task "msco" \
  --project_name "GDSG" \
  --wandb_logger_name "dense_cross3s6u_test_3s8u" \
  --diffusion_type "hybrid" \
  --do_test \
  --test_batch 1 \
  --parallel_sampling 8 \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/root/GDSG/data/msmu-co" \
  --training_split "3server/3s8u_60000samples_20240606133850.txt" \
  --validation_split "3server/3s8u_2000samples_20240606181517.txt" \
  --test_split "3server/3s8u_2000samples_20240606181517.txt" \
  --batch_size 64 \
  --num_epochs 200 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --diffusion_steps 200 \
  --inference_diffusion_steps 5 \
  --hidden_dim 256 \
  --n_layers 5 \
  --ckpt_path "/root/GDSG/data/msmu-co/models/msco_dense_hybrid_3s6u20240615172616/x6md5uyx/checkpoints/last.ckpt"