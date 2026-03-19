#!/bin/bash -l
#SBATCH --job-name=SF_Auto_Queue
#SBATCH --output=logs/sf_queue_%j.out
#SBATCH --error=logs/sf_queue_%j.err
#SBATCH --time=10:00:00  
#SBATCH --mem=32g
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

# Go to the root directory
cd /common/home/rvk22/spaceforge-xai-run5

for FILE_PATH in active_jobs/*.txt; do
    
    [ -e "$FILE_PATH" ] || { echo "No jobs found in active_jobs/"; break; }

    JOB_FILE=$(basename "$FILE_PATH")
    CURRENT_ID="${JOB_FILE%.txt}"
    
    echo "------------------------------------------------"
    echo "STARTING: $CURRENT_ID"
    echo "------------------------------------------------"

    # Navigate into Sim/ for the run command
    cd Sim/

    # Notice we REMOVED the INPUT_SUBDIR override.
    # We pass the relative path directly into --job-file.
    RUN_ID="$CURRENT_ID" \
    MODE=wake \
    ENABLE_SPARTA=ON \
    GPU=OFF \
    WAKE_DECK=in.wake_harness \
    NP=8 \
    ./run.sh --nticks 1350 --couple-every 1 --sparta-block 2500 --job-file "../active_jobs/$JOB_FILE"

    # Go back to root for the next loop iteration
    cd ..
done