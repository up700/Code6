#!/bin/bash
GPUID=$1
echo "Run on GPU $GPUID"
TRAIN=True
TEST=False
# data
DATASET=$2
SAMPLES=$3
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
DATA_ROOT=$PROJECT_ROOT/dataset/

# model
TOKENIZER_TYPE=roberta
SPAN_TYPE=roberta
TYPE_TYPE=roberta
TOKENIZER_NAME=xlm-roberta-large

MODEL_NAME=xlm-roberta-large

# params
LR=1e-5
WEIGHT_DECAY=0
EPOCH=30
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=400

TRAIN_BATCH=16
EVAL_BATCH=32


# output
OUTPUT=$PROJECT_ROOT/ptms-aug/$DATASET$SAMPLES/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u run_script.py --data_dir $DATA_ROOT \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --max_seq_length 128 \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --max_grad_norm 1.0 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --gradient_accumulation_steps 1 \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --seed $SEED \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --samples $SAMPLES \
  --lambda0 1.5 \
  --alpha 0.01 \
  --beta1 0.1 \
  --beta2 0.1 \
  --aug \
  --save_best \

#####################################################

GPUID=$1
echo "Run on GPU $GPUID"
TRAIN=True
TEST=False
# data
DATASET=$2
SAMPLES=$3
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
DATA_ROOT=$PROJECT_ROOT/dataset/

# model
TOKENIZER_TYPE=roberta
SPAN_TYPE=roberta
TYPE_TYPE=roberta
TOKENIZER_NAME=xlm-roberta-large

MODEL_NAME=$PROJECT_ROOT/ptms-aug/$DATASET$SAMPLES/checkpoint-best-dev/


# params
LR=1e-5
WEIGHT_DECAY=0
EPOCH=500
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=400

TRAIN_BATCH=16
EVAL_BATCH=32

# output
OUTPUT=$PROJECT_ROOT/ptms/$DATASET$SAMPLES/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u run_script.py --data_dir $DATA_ROOT \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --max_seq_length 128 \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --max_grad_norm 1.0 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --gradient_accumulation_steps 1 \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --seed $SEED \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --samples $SAMPLES \
  --lambda0 1.5 \
  --alpha 0.01 \
  --beta1 0.1 \
  --beta2 0.1 \
  --save_best \

#####################################################################

GPUID=$1
echo "Run on GPU $GPUID"
TRAIN=False
TEST=True
# data
DATASET=$2
SAMPLES=$3
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
DATA_ROOT=$PROJECT_ROOT/dataset/

# model
TOKENIZER_TYPE=roberta
SPAN_TYPE=roberta
TYPE_TYPE=roberta
TOKENIZER_NAME=xlm-roberta-large

MODEL_NAME=xlm-roberta-large

# params
LR=1e-5
WEIGHT_DECAY=0
EPOCH=1000
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=400

TRAIN_BATCH=16
EVAL_BATCH=32

# output
OUTPUT=$PROJECT_ROOT/ptms/$DATASET$SAMPLES/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u run_script.py --data_dir $DATA_ROOT \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --max_seq_length 128 \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --max_grad_norm 1.0 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --gradient_accumulation_steps 1 \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --seed $SEED \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --samples $SAMPLES \
  --lambda0 1.5 \
  --alpha 0.01 \
  --beta1 0.1 \
  --beta2 0.1 \
