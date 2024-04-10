#!/bin/bash

# Checking if user is root
if [ $(id -u) = 0 ]; then
    echo "* HITO 2 - HPC  *"
    echo "Authors: Carlos Pulido, Antonio Mateo"
else
    echo "User is not root!! Please be root and try again"
    exit 1
fi

# Setting up venv
mkdir -p venv
python3 -m venv ./venv
source venv/bin/activate

# Installing dependencies and modules
sudo apt update
sudo apt install libopenmpi-dev -y
pip install -r requirements.txt

# Definition of the K-Means tests characteristics
min_k=2
max_k=10
repetitions=5
sleep=5
mpi_processes=(4 8 12 24)
files=("MacQueen_V1.py" "MacQueen_V2.py" "MacQueenMPI_V1.py" "MacQueenMPI_V2.py")

# Loading pavia.txt text file to apply K-Means
echo "Executing load.py..."
python3 load.py
echo "Finished execution of load.py."

# Testing several times (now 5) each file set up previously
for file in "${files[@]}"
do
    if [[ $file == *"MPI"* ]]; then
        for ((k=min_k; k<=max_k; k++))
        do
            for num_processes in "${mpi_processes[@]}"
            do
                if [[ $num_processes -ne 24 ]]; then
                    sleep $sleep
                    echo "Executing $file with k=$k and p=$num_processes"
                    mpiexec -n $num_processes --allow-run-as-root python3 "$file" --k $k --distance_metric euclidean --max_iters 100 --centroid_tolerance 0.2 --image False --repetitions $repetitions
                    echo "Finished execution of $file with k=$k and p=$num_processes"
                else
                    sleep $sleep
                    echo "Executing $file with k=$k and p=$num_processes"
                    mpiexec -n $num_processes --allow-run-as-root --use-hwthread-cpus python3 "$file" --k $k --distance_metric euclidean --max_iters 100 --centroid_tolerance 0.2 --image False --repetitions $repetitions
                    echo "Finished execution of $file with k=$k and p=$num_processes"
                fi
            done
        done
    else
        for ((k=min_k; k<=max_k; k++))
        do
            sleep $sleep
            echo "Executing $file with k=$k..."
            python3 "$file" --k $k --distance_metric euclidean --max_iters 100 --centroid_tolerance 0.2 --image False --repetitions $repetitions
            echo "Finished execution of $file with k=$k."
        done
    fi
done

# Generate charts
python3 charts.py --directories kmeans/mpi/macqueenV1 kmeans/mpi/macqueenV2 --mpi True
python3 charts.py --directories kmeans/sequential/macqueenV1 kmeans/sequential/macqueenV2 --mpi False