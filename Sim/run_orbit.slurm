#!/bin/bash -l
#SBATCH --job-name=V2_Jobs30_orbit1
#SBATCH --output=logs/sf_orbit1_job30_%j.out
#SBATCH --error=logs/sf_orbit1_job30_%j.err
#SBATCH --time=100:00:00
#SBATCH --mem=32g
#SBATCH -c 8

cd /common/home/rvk22/spaceforge-xai-run3/Sim

RUN_ID=V2_Jobs30 MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=1 \
  ./run.sh --nticks 1220 --couple-every 1 --sparta-block 2500
