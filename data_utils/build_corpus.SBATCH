#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="build"
#SBATCH --output=log_test/%j_build_computers_corpus.out
#SBATCH --error=log_test/%j_build_computers_corpus.err
#SBATCH --account=pr_142_tandon_advanced

module purge

singularity exec --nv --overlay /scratch/jl11523/MyNewSing/overlay-25GB-500K.ext3:ro \
    /scratch/jl11523/MyNewSing/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "


source /ext3/env.sh


conda activate graphgpt_eval

python build_corpus.py

"

