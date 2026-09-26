#!/bin/bash
#SBATCH --job-name=sip # metto colonna NAME di squeue
#SBATCH --mail-type=ALL # qui lasciare così
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it

#SBATCH --partition=l40s
#SBATCH --nodelist=slurmwkrl40s3.disi.unibo.it # s; s1; s2; s3
#SBATCH --gres=gpu:l40s:2 # qui metto in base a quel che dice scontrol show nodes
#SBATCH --ntasks-per-node=1

#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

export APPTAINER_CACHEDIR=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection

# apptainer pull sip.sif docker://loriringhio97/tep_detectors:latest

# apptainer run --nv tesi.sif python3 sip_train.py --SEED 1 --TOKENIZER nf_msg_tokenizer --DATASET voip_nassar --N_EMB 512 --N_LAYER 8 --N_HEAD 8 --BATCH_SIZE 32 --LR 5e-4 --ARCH gpt --DROPOUT 0.1
# apptainer run --nv sip.sif python3 /scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection-main/main_train.py AE BATADAL --run_name autoencoder --ae_model_params_layers 5 --ae_model_params_cf 2

CONFIGS=(
"r1 4 128 50 3"
"r2 3 256 50 3"
"r3 4 256 50 3"
"r4 3 128 40 3"
)

for CFG in "${CONFIGS[@]}"; do

    read TAG LAYERS UNITS HISTORY KERNEL <<< "$CFG"
    RUN="cnn_${TAG}"

    mkdir -p models/${RUN}

    apptainer run --nv sip.sif python3 \
    /scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection/main_train.py \
    CNN BATADAL \
    --run_name ${RUN} \
    --seed 42 \
    --cnn_model_params_layers ${LAYERS} \
    --cnn_model_params_units ${UNITS} \
    --cnn_model_params_history ${HISTORY} \
    --cnn_model_params_kernel ${KERNEL} \
    --train_params_batch_size 256 \
    --train_params_epochs 150 \
    --detect_params_metrics recall \
    --detect_params_max_fpr 0.05 \
    --detect_params_windows \
    3 4 5 6 7 8 9 10 11 12 15 \
    --detect_params_percentile \
    0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 \
    0.86 0.87 0.88 0.89 0.90 0.91 0.92 0.93 \
    0.94 0.95 0.96 0.97 0.98 0.99

done