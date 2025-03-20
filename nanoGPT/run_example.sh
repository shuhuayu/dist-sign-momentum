cd fairscale
pip install -r requirements.txt
pip install -e .
cd ..
cd nanoGPT

# I/O
DATA_DIR=data/owt
OUT_DIR=${DATA_DIR}/checkpoint
EVAL_ITV=20
# Model, gpt2s
N_LAYER=12
N_HEAD=12
N_EMBD=768
# data
BSZ=6
GDAC=80
# LR
LR=0.0005
MIN_LR=0.000025
# WANDB
WAND_PROJ=openwebtext-gpt2s

# adamw ddp
# WAND_RUN=adamw_ddp
# torchrun --standalone --nnodes=1 --nproc_per_node=8 \
#     train_ddp.py --data-dir $DATA_DIR --out-dir $OUT_DIR --eval-interval $EVAL_ITV \
#     --wandb-log --wandb-project $WAND_PROJ --wandb-run-name $WAND_RUN \
#     --n-layer $N_LAYER --n-head $N_HEAD --n-embd $N_EMBD \
#     --batch-size $BSZ --gradient-accumulation-steps $GDAC \
#     --learning-rate $LR --decay-lr --min-lr $MIN_LR --use-oss

# slowmo
# WAND_RUN=slowmo
# torchrun --standalone --nnodes=1 --nproc_per_node=8 \
#     train_fastdt.py --data-dir $DATA_DIR --out-dir $OUT_DIR --eval-interval $EVAL_ITV \
#     --wandb-log --wandb-project $WAND_PROJ --wandb-run-name $WAND_RUN \
#     --n-layer $N_LAYER --n-head $N_HEAD --n-embd $N_EMBD \
#     --batch-size $BSZ --gradient-accumulation-steps $GDAC \
#     --learning-rate $LR --decay-lr --min-lr $MIN_LR \
#     --fastdt-momentum 0.2 --fastdt-gap-penalty 1.0 --fastdt-lr 1.0 --fastdt-frequency 12

# our proposed sign momentum
WAND_RUN=sign-mom
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    train_fastdt.py --data-dir $DATA_DIR --out-dir $OUT_DIR --eval-interval $EVAL_ITV \
    --wandb-log --wandb-project $WAND_PROJ --wandb-run-name $WAND_RUN \
    --n-layer $N_LAYER --n-head $N_HEAD --n-embd $N_EMBD \
    --batch-size $BSZ --gradient-accumulation-steps $GDAC \
    --learning-rate $LR --decay-lr --min-lr $MIN_LR \
    --fastdt-lr 1.0 --fastdt-frequency 12 --fastdt-use-lion