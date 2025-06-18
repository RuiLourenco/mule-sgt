#!/bin/bash

PID=4109978
INTERVAL=2

while true; do
    if [ -r /proc/$PID/status ]; then
        mem_gb=$(awk '/VmRSS/ {printf "%.4f", $2/1024/1024}' /proc/$PID/status)
        # Print with carriage return and no newline, flush output
        echo -ne "\r$(date '+%H:%M:%S') PID $PID Memory Usage: $mem_gb GB    "
    else
        echo -e "\n$(date '+%H:%M:%S') PID $PID no longer exists or cannot be read."
        exit 1
    fi
    sleep $INTERVAL
done