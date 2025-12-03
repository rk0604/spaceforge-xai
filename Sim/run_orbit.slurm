#!/bin/bash -l
#SBATCH --job-name=sf_orbit1
#SBATCH --output=logs/sf_orbit1_%j.out
#SBATCH --error=logs/sf_orbit1_%j.err
#SBATCH --time=45:00:00
#SBATCH --mem=32g
#SBATCH -c 8

cd /common/home/<netid>/spaceforge-xai/Sim

RUN_ID=orbit1 MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=1 \
  ./run.sh --nticks 50000 --couple-every 1 --sparta-block 2500
