#!/bin/bash

# Define the number of times you want to run the script
times=10

# Ensure results.txt is empty before starting
> results.txt

for (( i=1; i<=times; i++ ))
do
    echo "Run $i:" >> results.txt

    # Capture start time in nanoseconds
    start_time=$(date +%s%N)

    # Run the Python script and capture its output
    python3 run_command.py
    output=$(cat timing_stats_io.txt)
    # Extract Python start time from the output
    end_time=$(date +%s%N)
    python_start_time=$(echo "$output" | grep "Start Time:" | awk '{print $3}')

    # Capture end time in nanoseconds

    # Compute the whole duration (end time - start time) in nanoseconds, then convert to seconds
    duration=$((end_time - start_time))
    duration_in_seconds=$(echo "scale=9; $duration / 1000000000" | bc)

    # Compute the initialization duration (Python start time - Bash start time) in seconds
    # Make sure to correct this formula as per your requirement
    # If python_start_time is in seconds and you want the difference from start_time in Bash (in nanoseconds),
    # You need to convert start_time to seconds first before the subtraction
    init_time=$(echo "scale=9; $python_start_time - ($start_time) / 1000000000" | bc)
    echo $ouput >> results.txt
    echo "Initialization duration: $init_time seconds" >> results.txt
    echo "Whole duration: $duration_in_seconds seconds" >> results.txt
    echo "-----------------------" >> results.txt

done

