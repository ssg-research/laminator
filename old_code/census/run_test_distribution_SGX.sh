#!/bin/bash

# Define the number of times you want to run the script
times=10

# Initialize array with correct syntax
declare -a array
array[0]="race_x"
array[1]="sex_x"
array[2]="race_z-y"
array[3]="sex_z-y"

# Ensure results.txt is empty before starting
> results.txt

# Outer loop to go through different types of attestation
for (( j=0; j<=3; j++ ))
do
    echo "Testing ${array[j]}"
    echo "Testing ${array[j]}" >> results.txt

    # Inner loop to run the script multiple times
    for (( i=1; i<=times; i++ ))
    do
        echo "Run $i:"
        echo "Run $i:" >> results.txt

        # Capture start time in seconds (with nanosecond precision converted to seconds)
        start_time=$(date +%s.%N)

        # Run the Python script and capture its output including start time details
        output=$(gramine-sgx ./distribution distribution_attestation.py --attestation "${array[j]}" 2>&1)

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

    echo "Test executions for ${array[j]} completed."
    echo "-----------------------" >> results.txt
done

echo "All tests completed."
