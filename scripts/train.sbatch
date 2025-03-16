#! /bin/bash
#
# Copyright 2023 German Arutyunov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#SBATCH --mail-user=germanarutyunov@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=gt-train
#SBATCH --output="logs/slurm/train/"%j.out
#SBATCH --error="logs/slurm/train/"%j.err
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=5
#SBATCH --time=3-0:0

# Load modules
chmod +x ./prepare.sh
source ./prepare.sh

cd scripts

# Train model

function show_help {
    fairseq-train --help
    exit 0
}

# Default values
DATASET_NAME="typilus"
DATASET_ROOT="~/git-py"
MODEL_ARCH="tokengt_base"
MODEL_NAME=""
BATCH_SIZE=16
NUM_CLASSES=100
MAX_TOKENS=10000
PORT=1369
PROCESSED_DIR="processed-data"
USER_DIR="../graph_coder"
NUM_WORKERS="0"
DDP_BACKEND="pytorch_ddp"
TASK="deep_similarity_learning"
CRITERION="typilus_loss"
PERFORMER=true
PERFORMER_FEATURE_REDRAW_INTERVAL="100"
STOCHASTIC_DEPTH=true
PRENORM=true
ATTENTION_DROPOUT="0.0"
ACT_DROPOUT="0.1"
DROPOUT="0.0"
OPTIMIZER="adam"
WEIGHT_DECAY="0.001"
CLIP_NORM="25.0"
LR_SCHEDULER="polynomial_decay"
POWER="0.5"
WARMUP_UPDATES="3000"
TOTAL_NUM_UPDATE="50000"
LR="0.1"
END_LEARNING_RATE="1e-7"
DATA_BUFFER_SIZE="20"
NO_EPOCH_CHECKPOINTS=true
PATIENCE="10"
FIND_UNUSED_PARAMETERS=true
HELP=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --model-arch)
            MODEL_ARCH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --processed-dir)
            PROCESSED_DIR="$2"
            shift 2
            ;;
        --user-dir)
            USER_DIR="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --ddp-backend)
            DDP_BACKEND="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --criterion)
            CRITERION="$2"
            shift 2
            ;;
        --performer-feature-redraw-interval)
            PERFORMER_FEATURE_REDRAW_INTERVAL="$2"
            shift 2
            ;;
        --attention-dropout)
            ATTENTION_DROPOUT="$2"
            shift 2
            ;;
        --act-dropout)
            ACT_DROPOUT="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --clip-norm)
            CLIP_NORM="$2"
            shift 2
            ;;
        --lr-scheduler)
            LR_SCHEDULER="$2"
            shift 2
            ;;
        --power)
            POWER="$2"
            shift 2
            ;;
        --warmup-updates)
            WARMUP_UPDATES="$2"
            shift 2
            ;;
        --total-num-update)
            TOTAL_NUM_UPDATE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --end-learning-rate)
            END_LEARNING_RATE="$2"
            shift 2
            ;;
        --data-buffer-size)
            DATA_BUFFER_SIZE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-performer)
            PERFORMER=false
            shift
            ;;
        --no-stochastic-depth)
            STOCHASTIC_DEPTH=false
            shift
            ;;
        --no-prenorm)
            PRENORM=false
            shift
            ;;
        --epoch-checkpoints)
            NO_EPOCH_CHECKPOINTS=false
            shift
            ;;
        --no-find-unused-parameters)
            FIND_UNUSED_PARAMETERS=false
            shift
            ;;
        --help)
            HELP=true
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            # Pass any other parameters directly to the fairseq-train command
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        esac
    done

    # Pass any remaining arguments after -- directly to fairseq-train
    EXTRA_ARGS="$EXTRA_ARGS $@"

    # Show help if requested
    if [ "$HELP" = true ]; then
        show_help
    fi

    # If model name is not set, create it from dataset name and model arch
    if [ -z "$MODEL_NAME" ]; then
        MODEL_NAME="${DATASET_NAME}-${MODEL_ARCH}"
    fi

    NUM_DATA_WORKERS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS))

    echo "Training..."
    echo "Port: $PORT"
    echo "Model arch: $MODEL_ARCH"
    echo "Dataset name: $DATASET_NAME"
    echo "Dataset root: $DATASET_ROOT"
    echo "Max tokens: $MAX_TOKENS"
    echo "Saving checkpoints to: ckpts/$MODEL_NAME"
    echo "Batch size: $BATCH_SIZE"
    echo "Num classes: $NUM_CLASSES"
    echo "Extra arguments: $EXTRA_ARGS"

    ulimit -c unlimited

    # Create the command and store it in a variable
    TRAIN_CMD="fairseq-train \\
    --user-dir \"$USER_DIR\" \\
    --num-workers \"$NUM_WORKERS\" \\
    --ddp-backend=\"$DDP_BACKEND\" \\
    --task \"$TASK\" \\
    --criterion \"$CRITERION\" \\
    $([ "$PERFORMER" = true ] && echo "--performer") \\
    --performer-feature-redraw-interval \"$PERFORMER_FEATURE_REDRAW_INTERVAL\" \\
    $([ "$STOCHASTIC_DEPTH" = true ] && echo "--stochastic-depth") \\
    $([ "$PRENORM" = true ] && echo "--prenorm") \\
    --attention-dropout \"$ATTENTION_DROPOUT\" --act-dropout \"$ACT_DROPOUT\" --dropout \"$DROPOUT\" \\
    --optimizer \"$OPTIMIZER\" --weight-decay \"$WEIGHT_DECAY\" \\
    --clip-norm \"$CLIP_NORM\" \\
    --lr-scheduler \"$LR_SCHEDULER\" --power \"$POWER\" --warmup-updates \"$WARMUP_UPDATES\" --total-num-update \"$TOTAL_NUM_UPDATE\" \\
    --lr \"$LR\" --end-learning-rate \"$END_LEARNING_RATE\" \\
    --data-buffer-size \"$DATA_BUFFER_SIZE\" \\
    $([ "$NO_EPOCH_CHECKPOINTS" = true ] && echo "--no-epoch-checkpoints") \\
    --patience \"$PATIENCE\" \\
    $([ "$FIND_UNUSED_PARAMETERS" = true ] && echo "--find-unused-parameters") \\
    --dataset-name \"$DATASET_NAME\" \\
    --dataset-root \"$DATASET_ROOT\" \\
    --arch \"$MODEL_ARCH\" \\
    --save-dir ./ckpts/\"$MODEL_NAME\" \\
    --tensorboard-logdir ./tb/\"$MODEL_NAME\" \\
    --batch-size \"$BATCH_SIZE\" \\
    --num-classes \"$NUM_CLASSES\" \\
    --max-tokens \"$MAX_TOKENS\" \\
    --distributed-port \"$PORT\" \\
    --processed-dir \"$PROCESSED_DIR\" \\
    --num-data-workers $NUM_DATA_WORKERS \\
    $EXTRA_ARGS"

# Echo the command
echo "Executing command:"
echo "$TRAIN_CMD"

# Execute the command only if not in dry run mode
if [ "$DRY_RUN" = false ]; then
    eval "$TRAIN_CMD"
else
    echo "Dry run mode - command not executed"
fi
