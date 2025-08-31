#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --job-name=convert_swissai_to_apertus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00

usage() {
        echo "Usage: sbatch convert_swissai_to_apertus.sbatch <swissai-model-path> <apertus-output-path>"
}
die() {
        echo "$*" >& 2
        exit 1
}

set -e
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

if (( $# < 2 )); then
        usage
        die "Invalid usage: At least 2 arguments required"
fi

SWISSAI_MODEL_PATH=$1
APERTUS_OUTPUT_PATH=$2
PATH_TO_TOKENIZER=${3:-/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/tokenizer}
IS_BASE=${IS_BASE:-0}
PATH_TO_HFCONVERTER=${PATH_TO_HFCONVERTER:-/iopsstor/scratch/cscs/ansaripo/hfconverter}
TRANSFORMERS_BRANCH=${TRANSFORMERS_BRANCH:-"swissai-apertus-convert"}
FORCE=${FORCE:-0}

if [ "$PATH_TO_TOKENIZER" != "None" ]; then
    COMMAND="python convert_swissai_to_apertus.py \"$SWISSAI_MODEL_PATH\" \"$APERTUS_OUTPUT_PATH\" --path-to-tokenizer \"$PATH_TO_TOKENIZER\""
else
    COMMAND="python convert_swissai_to_apertus.py \"$SWISSAI_MODEL_PATH\" \"$APERTUS_OUTPUT_PATH\""
fi

if [ "$IS_BASE" == "1" ]; then
    COMMAND="$COMMAND --is-base"
fi

if [ "$FORCE" == "1" ]; then
    COMMAND="$COMMAND --force"
fi

echo "Command to run: $COMMAND"

mkdir -p logs

mkdir -p $APERTUS_OUTPUT_PATH


echo "Running SwissAI to Apertus conversion..."
echo "SwissAI model path: $SWISSAI_MODEL_PATH"
echo "Apertus output path: $APERTUS_OUTPUT_PATH"
echo "Path to tokenizer: $PATH_TO_TOKENIZER"
echo "Path to hfconverter: $PATH_TO_HFCONVERTER"
echo "Installing transformers from remote branch: $TRANSFORMERS_BRANCH"

srun -ul --environment=./env.toml bash -c "
    # Install the local transformers library
    pip install -v --no-cache-dir --no-build-isolation --no-deps --force-reinstall -U \
    \"transformers @ git+https://github.com/swiss-ai/transformers.git@${TRANSFORMERS_BRANCH}\" \
    'huggingface-hub>=0.34.0,<1.0' 'accelerate' 'safetensors>=0.4.1' 'shutils'

    cd $PATH_TO_HFCONVERTER
    $COMMAND
"

# Goodbye.
echo "Conversion completed."
echo "Output saved to: $APERTUS_OUTPUT_PATH"
echo "Goodbye."
echo "END TIME: $(date)"
