#!/bin/bash

# K-Means test characteristics
min_k=2
max_k=10
repetitions=5
sleep_time=5
openmp_threads=(4 8 12 24)
files=("macqueen_OpenMP_V2" "macqueen_OpenMP_V1" "macqueen_OpenMP_V2")

# Load pavia.txt to apply K-Means
echo "Executing load.py..."
python3 load.py
echo "Finished executing load.py."

# Build the executables
make clean
make

# Test each file with K-Means and OpenMP configurations
for file in "${files[@]}"; do
    for ((k = min_k; k <= max_k; k++)); do
        if [[ $file == *"OpenMP"* ]]; then
            # Test OpenMP variants with different thread counts
            for num_threads in "${openmp_threads[@]}"; do
                sleep $sleep_time
                echo "Executing $file with k=$k and threads=$num_threads..."
                OMP_NUM_THREADS=$num_threads ./"$file" $k 0 100 0.2 5 0
                echo "Finished execution of $file with k=$k and threads=$num_threads."
            done
        else
            # Test non-OpenMP variants
            sleep $sleep_time
            echo "Executing $file with k=$k..."
            ./"$file" --k $k
            echo "Finished execution of $file with k=$k."
        fi
    done
done
