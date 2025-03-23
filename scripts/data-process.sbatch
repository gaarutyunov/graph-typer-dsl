#!/bin/bash
#SBATCH --job-name=gt-data           # Название задачи
#SBATCH --output="logs/slurm/data/"%j.out
#SBATCH --error="logs/slurm/data/"%j.err
#SBATCH --time=3-0:00               # Максимальное время выполнения
#SBATCH --ntasks=1                   # Количество MPI процессов
#SBATCH --nodes=1                    # Требуемое кол-во узлов
#SBATCH --cpus-per-task=10            # Требуемое кол-во CPU

# Default values
DATASET_ROOT="~/git-py"
SPLIT="train"
MAX_TOKENS=10000
NUM_CLASSES=100
PROCESSED_DIR=""
NUM_DATA_WORKERS="$SLURM_CPUS_PER_TASK"
HELP=false
DRY_RUN=false

# Parse named arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            HELP=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --dataset-root=*)
            DATASET_ROOT="${1#*=}"
            ;;
        --split=*)
            SPLIT="${1#*=}"
            ;;
        --max-tokens=*)
            MAX_TOKENS="${1#*=}"
            ;;
        --processed-dir=*)
            PROCESSED_DIR="${1#*=}"
            ;;
        --num-workers=*)
            NUM_DATA_WORKERS="${1#*=}"
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Show help if requested
if [ "$HELP" = true ]; then
    PYTHONPATH=. python -m graph_coder.data --help
    exit 0
fi

# Set processed dir default if not provided
if [ -z "$PROCESSED_DIR" ]; then
    PROCESSED_DIR="processed-data-$MAX_TOKENS"
fi

echo "Processing dataset $DATASET_NAME"
echo "Root: $DATASET_ROOT"
echo "Processed dir: $PROCESSED_DIR"
echo "Split: $SPLIT"
echo "Max tokens: $MAX_TOKENS"
echo "Num workers: $NUM_DATA_WORKERS"

COMMAND="PYTHONPATH=. python -m graph_coder.data \
--dataset-root \"$DATASET_ROOT\" \
--processed-dir \"$PROCESSED_DIR\" \
--split \"$SPLIT\" \
--max-tokens \"$MAX_TOKENS\" \
--num-workers \"$NUM_DATA_WORKERS\""

if [ "$DRY_RUN" = true ]; then
    echo "Command (dry run):"
    echo "$COMMAND"
else
    eval "$COMMAND"
fi
