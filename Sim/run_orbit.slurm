#!/bin/bash -l
#SBATCH --job-name=V3_Job2_orbit1
#SBATCH --output=logs/sf_orbit1_v3_job2_%j.out
#SBATCH --error=logs/sf_orbit1_v3_job2_%j.err
#SBATCH --time=100:00:00
#SBATCH --mem=32g
#SBATCH -c 8

cd /common/home/rvk22/spaceforge-xai-run3/Sim

RUN_ID=V3_Job2 MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=1 \
  ./run.sh --nticks 1220 --couple-every 1 --sparta-block 2500
