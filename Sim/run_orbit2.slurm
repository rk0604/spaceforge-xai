#!/bin/bash -l
#SBATCH --job-name=V3_Job24_orbit1_re_run
#SBATCH --output=logs/sf_orbit1_v3_job24re_run_%j.out
#SBATCH --error=logs/sf_orbit1_v3_job24re_run_%j.err
#SBATCH --time=100:00:00
#SBATCH --mem=32g
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

cd /common/home/rvk22/spaceforge-xai-run3/Sim

RUN_ID=V4_testing_2 MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=8 \
  ./run.sh --nticks 1250 --couple-every 1 --sparta-block 2500