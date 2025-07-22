#!/bin/bash

# Define the decoder path
BASE_DIR="$(dirname "$(dirname "$(pwd)")")"
echo "Base directory: $BASE_DIR"
DECODER="$BASE_DIR/build/main/bin/MSGTDecoder"

# Define the rates
RATES=(0.75 0.1 0.02 0.005)

# Loop through each rate and run the decoder
for RATE in "${RATES[@]}"; do
    OUTPUT_DIR="./${RATE}/"
    INPUT_FILE="danger_${RATE}.comp"
    # Create the output directory if it does not exist
    mkdir -p "$OUTPUT_DIR"
    echo "Created directory: $(realpath "$OUTPUT_DIR")"
    
    # Run the decoder with the specified flags
    $DECODER -v 13 13 -b 1 1 --lenslet13x13 --extension-repeat --t_gain 1 --bt601 -V -o "$OUTPUT_DIR" -i "$INPUT_FILE"
done