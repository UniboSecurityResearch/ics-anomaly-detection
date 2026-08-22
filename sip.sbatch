#!/bin/bash
#SBATCH --job-name=sip # metto colonna NAME di squeue
#SBATCH --mail-type=ALL # qui lasciare così
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --partition=l40s
#SBATCH --nodelist=slurmwkrl40s2.disi.unibo.it # s; s1; s2; s3
#SBATCH --exclusive          # uso esclusivo del nodo; se qualcuno lo sta usando tolgo o uso un altro nodo
#SBATCH --gres=gpu:l40s:4 # qui metto in base a quel che dice scontrol show nodes
#SBATCH --ntasks-per-node=224
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection/
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

export APPTAINER_CACHEDIR=/scratch.hpc/dhyana.schirra/tesi

# apptainer pull sip.sif docker://loriringhio97/tep_detectors:latest

apptainer run --nv sip.sif python3 /scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection/main_train.py GRU BATADAL --run_name gru --gru_model_params_layers 2 --gru_model_params_units 64 --gru_model_params_history 50