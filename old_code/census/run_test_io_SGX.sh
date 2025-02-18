#!/bin/bash

# Define the number of times you want to run the script
times=10

# Ensure results.txt is empty before starting
> results.txt

for (( i=1; i<=times; i++ ))
do
    echo "Run $i:" >> results.txt

    # Capture start time in seconds (with nanosecond precision converted to seconds)
    start_time=$(date +%s.%N)

    # Run the Python script and capture its output including start time details
    output=$(gramine-sgx ./io io.py --is_sgx y 2>&1)

    # Capture end time in seconds (with nanosecond precision converted to seconds)
    end_time=$(date +%s.%N)

    # Extract Python start time from the output
    python_start_time=$(echo "$output" | grep "Start Time:" | awk '{print $3}')

    # Compute the whole duration (end time - start time) in seconds using bc
    duration_in_seconds=$(echo "$end_time - $start_time" | bc)

    # Compute the initialization duration (Python start time - Bash start time) in seconds using bc
    init_duration_in_seconds=$(echo "$python_start_time - $start_time" | bc)

    # Log the output from the Python script and the computed durations
    echo "$output" >> results.txt
    echo "Initialization duration: $init_duration_in_seconds seconds" >> results.txt
    echo "Whole duration: $duration_in_seconds seconds" >> results.txt
    echo "-----------------------" >> results.txt

done

echo "Test executions completed."
