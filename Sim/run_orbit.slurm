#!/bin/bash -l
#SBATCH --job-name=V2_job30_RK
#SBATCH --output=logs/sf_orbit30_%j.out
#SBATCH --error=logs/sf_orbit30_%j.err
#SBATCH --time=100:00:00
#SBATCH --mem=32g
#SBATCH -c 8

cd /common/home/rvk22/spaceforge-xai/Sim

RUN_ID=V2_jobs30 MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=1 \
  ./run.sh --nticks 1220 --couple-every 1 --sparta-block 2500
