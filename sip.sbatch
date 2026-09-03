#!/bin/bash
#SBATCH --job-name=sip # metto colonna NAME di squeue
#SBATCH --mail-type=ALL # qui lasciare così
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --partition=l40s
#SBATCH --nodelist=slurmwkrl40s3.disi.unibo.it # s; s1; s2; s3
#SBATCH --exclusive          # uso esclusivo del nodo; se qualcuno lo sta usando tolgo o uso un altro nodo
#SBATCH --gres=gpu:l40s:2 # qui metto in base a quel che dice scontrol show nodes
#SBATCH --ntasks-per-node=224
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

export APPTAINER_CACHEDIR=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection

# apptainer pull sip.sif docker://loriringhio97/tep_detectors:latest

# apptainer run --nv tesi.sif python3 sip_train.py --SEED 1 --TOKENIZER nf_msg_tokenizer --DATASET voip_nassar --N_EMB 512 --N_LAYER 8 --N_HEAD 8 --BATCH_SIZE 32 --LR 5e-4 --ARCH gpt --DROPOUT 0.1
# apptainer run --nv sip.sif python3 /scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection-main/main_train.py AE BATADAL --run_name autoencoder --ae_model_params_layers 5 --ae_model_params_cf 2

apptainer run --nv sip.sif python3 /scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection/main_adversarial.py \
    --dataset BATADAL --model-type CNN --history 50 \
    --model-path models/cnn/CNN-BATADAL-l2-hist50-kern3-units64.h5 \
    --instance-threshold-percentile 0.995 --detection-window 5 \
    --sanity-check