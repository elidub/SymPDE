#!/bin/bash

cd ~/thesis/SymPDE/jobs/job_arrays

# List all jobs in the current directory
for job in $(ls *.job); do
    echo "Running $job"
    sbatch "$job"
done
