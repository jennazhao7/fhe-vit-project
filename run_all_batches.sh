#!/bin/bash

mkdir -p logs

# Launch 20 parallel jobs: each does 10 samples
for ((i=0; i<200; i+=10)); do
    end=$((i + 10))
    echo "Launching test_example $i to $end"
    ./build/test_example $i $end > logs/out_$i.txt &
done

wait
echo "All batches finished."