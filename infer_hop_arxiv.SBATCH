#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="arx-tq"
#SBATCH --output=log_titlequery/%j_infer_arxiv_32b_hop_similarity_titlequery.out
#SBATCH --error=log_titlequery/%j_infer_arxiv_32b_hop_similarity_titlequery.err
#SBATCH --account=pr_142_tandon_advanced

module purge

singularity exec --nv --overlay /scratch/jl11523/MyNewSing/overlay-25GB-500K.ext3:ro \
    /scratch/jl11523/MyNewSing/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "


source /ext3/env.sh



DATASET=arxiv
FILTER=Hop+PPR
RANKER=SemQA
CORPUS=./dataset/\${DATASET}/\${DATASET}_corpus.json
OUTPUT=./results/\${DATASET}_32b_\${FILTER}_\${RANKER}_titlequery.json

conda activate retriever
echo 'retriever start'

python retriever/graph_retriever_server.py  --corpus_path \$CORPUS \
                                            --topk 3 \
                                            --faiss_gpu \
                                            &

sleep 10
echo 'retriever OK'
conda deactivate


conda activate searchr1

python infer.py \
    --model_id /vast/jl11523/projects-local-models/Qwen2.5-32B-Instruct \
    --dataset \$DATASET \
    --output_path \$OUTPUT \
    --candidate_filter \$FILTER \
    --ranker \$RANKER
    
python cal_metric.py \
    --dataset \$DATASET \
    --result_file \$OUTPUT

"

#--model_id /vast/jl11523/projects-local-models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo
