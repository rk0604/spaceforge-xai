#!/bin/bash -l
#SBATCH --job-name=jobs7
#SBATCH --output=logs/sf_orbit1_%j.out
#SBATCH --error=logs/sf_orbit1_%j.err
#SBATCH --time=36:00:00
#SBATCH --mem=32g
#SBATCH -c 8

cd /common/home/rvk22/spaceforge-xai/Sim

RUN_ID=jobs7 MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=1 \
  ./run.sh --nticks 630 --couple-every 1 --sparta-block 2500
