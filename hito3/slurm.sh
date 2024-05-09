#!/bin/bash
#SBATCH --job-name=kmeans_job
#SBATCH --cpus-per-task=12

OMP_NUM_THREADS=12

if [ "$#" -ne 1 ]; then
    echo "Use: $0 <executable>"
    exit 1
fi

OMP_NUM_THREADS=$OMP_NUM_THREADS ./$1