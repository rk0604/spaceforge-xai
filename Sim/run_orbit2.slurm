#!/bin/bash -l
#SBATCH --job-name=V4_JOB_1_orbit1
#SBATCH --output=logs/sf_orbit1_V4_JOB_1%j.out
#SBATCH --error=logs/sf_orbit1_V4_JOB_1%j.err
#SBATCH --time=15:00:00
#SBATCH --mem=32g
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

cd /common/home/rvk22/spaceforge-xai-run4/Sim

RUN_ID=V4_TEST MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=8 \
  ./run.sh --nticks 250 --couple-every 1 --sparta-block 2500