#!/bin/bash

#SBATCH --job-name="NVE_MD_SPCE"
#SBATCH --output="NVE_MD_SPCE_%j.out"
#SBATCH --error="NVE_MD_SPCE_%j.err"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --mem=32G  # Specify memory requirement

# Set strict error handling
set -euo pipefail

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"

# Load required modules
module purge
module load gnu12/12.2.0
module load openmpi4/4.1.5

# Set environment variables
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Define variables
LAMMPS_EXEC="/home/fs01/om235/src/lammps-stable_29Sep2021/build/lmp_expanse"
INPUT_FILE="in.lammps"

# Check if LAMMPS executable exists
if [ ! -f "$LAMMPS_EXEC" ]; then
    echo "Error: LAMMPS executable not found at $LAMMPS_EXEC"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Run simulation
echo "Starting LAMMPS simulation..."
"$LAMMPS_EXEC" -in "$INPUT_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Simulation completed successfully."
else
    echo "Error: Simulation failed."
    exit 1
fi

echo "Job finished on $(date)"
