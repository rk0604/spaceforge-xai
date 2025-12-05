#!/bin/bash -l
#SBATCH --job-name=raunak_job
#SBATCH --output=logs/sf_orbit1_%j.out
#SBATCH --error=logs/sf_orbit1_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32g
#SBATCH -c 8

cd /common/home/ssb196/spaceforge-xai/Sim

RUN_ID=raunak_job MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=1 \
  ./run.sh --nticks 1510 --couple-every 1 --sparta-block 2500
