#!/bin/bash
#SBATCH --job-name=Eval_Sensitivity
#SBATCH --output=/project/logs/eval_sens_%j.out
#SBATCH --error=/project/logs/eval_sens_%j.err
#SBATCH --time=25:00:00
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH -N 1

set -euo pipefail

# ==========================================
# 1. Environment Setup
# ==========================================
# Load Conda
module load Miniconda3/23.9.0-0

# Initialize Conda in non-interactive mode
eval "$(conda shell.bash hook)"
conda activate /project/ishtyaq/genomic_ishtyaq_env

PYTHON_EXEC="/project/ishtyaq/genomic_ishtyaq_env/bin/python"
ORIGINAL_SCRIPT="/project//evaluation.py"

# Base directories
BASE_INPUT="/project/output"
BASE_OUTPUT="/project//Sensitivity_analaysis_output"

# Spatial-neighborhood/niche-size sensitivity conditions. These names must
# exactly match HN_RUN_NAME values in run_controlled_hyperniche.sh.
CONDITIONS=(
    "spatialk8_min4_lambda0p025"
    "spatialk12_min4_lambda0p025"
    "spatialk20_min4_lambda0p025"
)

echo "=========================================================="
echo " Starting Automated Sensitivity Evaluations"
echo "=========================================================="

for COND in "${CONDITIONS[@]}"; do
    echo ""
    echo ">>> Processing Condition: ${COND} <<<"
    
    # Define exact paths for this specific condition
    TARGET_INPUT="${BASE_INPUT}/${COND}"
    TARGET_OUTPUT="${BASE_OUTPUT}/${COND}_results"
    
    # Create the output directory
    mkdir -p "${TARGET_OUTPUT}"
    
    # Create a temporary python script for this iteration
    TEMP_SCRIPT="temp_eval_${COND}.py"
    cp "${ORIGINAL_SCRIPT}" "${TEMP_SCRIPT}"
    
    # Dynamically replace the hardcoded paths using 'sed'
    # The '|' character is used as a delimiter so path slashes '/' don't break the command
    sed -i "s|INPUT_DIR = .*|INPUT_DIR = \"${TARGET_INPUT}\"|g" "${TEMP_SCRIPT}"
    sed -i "s|OUTPUT_DIR = .*|OUTPUT_DIR = \"${TARGET_OUTPUT}\"|g" "${TEMP_SCRIPT}"
    
    # Execute the temporary script
    ${PYTHON_EXEC} -u "${TEMP_SCRIPT}"
    
    # Clean up the temporary file
    rm "${TEMP_SCRIPT}"
    
    echo "✅ Finished evaluating ${COND}. Results saved to ${TARGET_OUTPUT}"
done

echo ""
echo "🎉 All sensitivity evaluations completed successfully!"
