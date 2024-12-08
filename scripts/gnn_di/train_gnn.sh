export PYTHONPATH="$PWD..\difusco:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u gnn_run.py \
  --task "msco" \
  --wandb_logger_name "msco_dense_gnndi_train" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "D:\Codes\GDSG\data\msmu-co" \
  --training_split "4server\4s12u_1000samples_20240428223926.txt" \
  --validation_split "4server\4s12u_1000samples_20240428223926.txt" \
  --test_split "4server\4s12u_1000samples_20240428223926.txt" \
  --batch_size 256 \
  --num_epochs 100 \
  --validation_examples 8 \
  --hidden_dim 128 \
  --n_layers 4