#!/bin/bash

BASE_DIR="$(dirname "$(dirname "$(dirname "$(pwd)")")")"
ENCODER_PATH="$BASE_DIR/build/new/bin/MSGTEncoder"
DECODER_PATH="$BASE_DIR/build/new/bin/MSGTDecoder"

LOG_FILE="process_log.txt"
echo "Process Log" > "$LOG_FILE"
echo "===================" >> "$LOG_FILE"

RATES=(0.005 0.02 0.1 0.75)

for RATE in "${RATES[@]}"; do
    CONFIG_ENCODE="SIDEBOARD_${RATE}_encode.conf"
    CONFIG_DECODE="SIDEBOARD_${RATE}_decode.conf"
    OUTPUT_COMP="./${RATE}/${CONFIG_ENCODE%.conf}.comp"
    
    echo "Processing rate: $RATE"
    START_TIME=$(date +%s)
    echo "Start time for rate $RATE: $(date)" >> "$LOG_FILE"

    echo "Running encoder for ${RATE}..."
    mkdir -p ./${RATE}/
    $ENCODER_PATH -c "$CONFIG_ENCODE"
    
    echo "Running decoder for ${RATE}..."
    $DECODER_PATH -c "$CONFIG_DECODE"

    END_TIME=$(date +%s)
    echo "End time for rate $RATE: $(date)" >> "$LOG_FILE"
    DURATION=$((END_TIME - START_TIME))
    DURATION_HMS=$(printf '%02d:%02d:%02d' $((DURATION/3600)) $(((DURATION%3600)/60)) $((DURATION%60)))
    echo "Duration for rate $RATE: ${DURATION_HMS}" >> "$LOG_FILE"
    echo "-------------------" >> "$LOG_FILE"
done
