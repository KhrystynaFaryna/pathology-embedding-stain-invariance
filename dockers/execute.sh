#!/bin/bash
#ls

#echo Entering in $1
cd $1
echo Running in $1


LOG_DIR=/data/pathology/projects/autoaugmentation/from_chansey_review/invariant/dashboard_logs #../../outputs/dashboard_logs
mkdir -p $LOG_DIR

LOG_PATH=$LOG_DIR/$SLURM_JOB_ID.log
echo Saving the log file in $LOG_PATH

#nvidia-smi >> $LOG_PATH
#env >> $LOG_PATH

# ls directory for higher chance of accurate filesystem
#ls
#ls config
#ls models
#ls ../Analysis
#ls ../Preprocessing

# git pull
#git status >> $LOG_PATH
#git status
#git pull

# Weights & Biases login
wandb login 1a37c24f73aac5543306c631e10ad6d227a9724b


echo "----------"
# /usr/bin/python3.6 "${@:2}" --JOB-ID=$SLURM_JOB_ID >> $LOG_PATH
#if [ $2 = "sweep" ]
#then
#    # launch a Weights & Biases sweep
#    echo Running Weights and Biases sweep "${@:2}"
#    #wandb agent diag_prostate/lesion_classification/$3 | tee -a $LOG_PATH
#else
    # normal operation
python3 "${@:2}" #--JOB-ID=$SLURM_JOB_ID | tee -a $LOG_PATH
#fi

#nvidia-smi >> $SLURM_JOB_ID.log
#env >> $SLURM_JOB_ID.log

#echo Executing python3.6 $2 $3 $4 $5 $6 $7 
#/usr/bin/python3.6 $2 $3 $4 $5 $6 $7 >> $SLURM_JOB_ID.log
# /usr/bin/python3.6 $2 | tee -a $LOG_PATH
#echo Executing python3.8 "${@:2}"
# /usr/bin/python3.6 "${@:2}" --JOB-ID=$SLURM_JOB_ID >> $LOG_PATH
#/usr/bin/python3.8 "${@:2}" 


#python3 pathology/users/khrystyna/code/tpu-master/models/hello.py
# Symlink libraries
#
#cd ~/source
#ln -s /mnt/netcache/pathology/projects/autoaugmentation/DigitalPathology DigitalPathology
#ln -s /mnt/netcache/pathology/projects/autoaugmentation/deeplearning deeplearning
#ln -s /mnt/netcache/pathology/projects/autoaugmentation/baseline baseline

# Set environment variables that SHOULD be set by the system.
#
#export CPLUS_INCLUDE_PATH="/usr/local/cuda-8.0/targets/x86_64-linux/include/"
#export MKL_THREADING_LAYER="GNU"

# Add NVIDIA libraries to the search path
#
#LIBRARY_PATH="/usr/local/cuda/lib64/stubs:"

# Add Anaconda to the PATH
#
#PATH="$HOME/anaconda/bin:$PATH"

# Update the custom packages to the python search path
#
#export PYTHONPATH="/opt/ASAP/bin:$HOME/source/DigitalPathology:$HOME/source/deeplearning:$HOME"

# Set working directory
#
#cd baseline
#echo "Working from directory:"
#pwd

# Print the command for logging.
#
#echo "Execute command: [${@}]"

# Execute the passed command.
#
#${@}

# Support kill signals
#
#asyncRun() {
#    "$@" &
#    pid="$!"
#    trap "echo 'Stopping PID $pid'; kill -SIGINT $pid" SIGINT SIGTERM
#
#    # A signal emitted while waiting will make the wait command return code > 128
#    # Let's wrap it in a loop that doesn't end before the process is indeed stopped
#    while kill -0 $pid > /dev/null 2>&1; do
#        wait
#    done
#}
#echo "Using async run"
#asyncRun $@

# With exec the new command takes control of this shell instead of creating a new one (useful for SIGINT).
#echo "Using exec"
#exec $@