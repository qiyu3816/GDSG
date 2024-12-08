export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u msco_train.py \
  --task "msco" \
  --project_name "GDSG" \
  --wandb_logger_name "msco_sparse_hybrid_20s61u" \
  --diffusion_type "hybrid" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/root/GDSG/data/msmu-co" \
  --training_split "20server/20s61u_80000samples_20240605093253.txt" \
  --validation_split "20server/20s61u_2000samples_20240604223706.txt" \
  --test_split "20server/20s61u_2000samples_20240604223706.txt" \
  --alternate_split "4server/4s10u_80000samples_20240605234002.txt" \
  --alternate_step 4 \
  --batch_size 128 \
  --num_epochs 60 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --diffusion_steps 100 \
  --inference_diffusion_steps 5 \
  --hidden_dim 256 \
  --n_layers 8 \
  --sparse