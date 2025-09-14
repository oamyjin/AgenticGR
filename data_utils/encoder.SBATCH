#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="encoder"
#SBATCH --output=log_test/%j_encoder_semantic_neighbors.out
#SBATCH --error=log_test/%j_encoder_semantic_neighbors.err
#SBATCH --account=pr_142_tandon_advanced

module purge

singularity exec --nv --overlay /scratch/jl11523/MyNewSing/overlay-25GB-500K.ext3:ro \
    /scratch/jl11523/MyNewSing/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "


source /ext3/env.sh


conda activate retriever

python encoder.py --dataset arxiv
python encoder.py --dataset products
python encoder.py --dataset pubmed
python encoder.py --dataset sports
python encoder.py --dataset computers

"

