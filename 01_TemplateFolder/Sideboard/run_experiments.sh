#!/bin/bash

# Log file to record start and end times
LOG_FILE="process_log.txt"

# Rates to process
RATES=("0.75" "0.1" "0.02" "0.005")

# Path to the MSGTEncoder binary
BASE_DIR="$(dirname "$(dirname "$(pwd)")")"
echo "Base directory: $BASE_DIR"
ENCODER_PATH="$BASE_DIR/build/main/bin/MSGTEncoder"

Clear the log file
echo "Process Log" > "$LOG_FILE"
echo "===================" >> "$LOG_FILE"

# Loop through each rate and run the encoder
for RATE in "${RATES[@]}"; do
    DIR_NAME=$(basename "$(pwd)")
    CONFIG_FILE="${DIR_NAME^^}_${RATE}_encode.conf"
    
    echo "Processing rate: $RATE"
    START_TIME=$(date +%s)
    echo "Start time for rate $RATE: $(date)" >> "$LOG_FILE"
    
    # Run the encoder
    $ENCODER_PATH -c "$CONFIG_FILE"
    
    END_TIME=$(date +%s)
    echo "End time for rate $RATE: $(date)" >> "$LOG_FILE"
    
    DURATION=$((END_TIME - START_TIME))
    DURATION_HMS=$(printf '%02d:%02d:%02d' $((DURATION/3600)) $(((DURATION%3600)/60)) $((DURATION%60)))
    echo "Duration for rate $RATE: ${DURATION_HMS}" >> "$LOG_FILE"
    
    echo "-------------------" >> "$LOG_FILE"
done


# Run the decoder script
./run_decoder.sh

# Run the MATLAB script
matlab -nodisplay -nojvm -r "cd('$(pwd)');run('eval/metrics_QM_$(basename $(pwd)).m'); exit"

# Change directory to eval
cd eval

# Run the Python script to plot CSV
python plot_csv.py

echo "All processes completed. Log saved to $LOG_FILE."