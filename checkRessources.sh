#!/bin/bash

INTERVAL=900
END=63
THRESHOLD=367
script=$1

while true; do
    job_count=$(squeue -u $USER | wc -l)
    num_models=$(ls Res/"${script%.py}" | wc -l)
    if (( num_models < END )); then
        if (( job_count < THRESHOLD )); then
            ./submitJobs.sh $((num_models+1)) $((num_models+THRESHOLD-job_count)) "$script"
        fi
    else
        break
    fi
    sleep $INTERVAL
done