#!/bin/bash

# Define the number of times you want to run the script
times=10

# Ensure results.txt is empty before starting
> base_results.txt

declare -a datasets=("CENSUS" "IMDB" "UTKFACE")
declare -a model_sizes=("VGG11" "VGG13" "VGG16" "VGG19")
declare -a attestations=("train" "accuracy" "io" "distribution")

echo "Starting tests"

for model_size in "${model_sizes[@]}"; do
    echo "MODEL SIZE: $model_size" >> base_results.txt

    for dataset in "${datasets[@]}"; do
        echo "DATASET: $dataset" >> base_results.txt
        for attestation in "${attestations[@]}"; do
            echo "ATTESTATION: $attestation" >> base_results.txt
            if [[ "$dataset" == "IMDB" && "$attestation" == "distribution" ]]; then
                continue
            fi


            for (( i=1; i<=times; i++ )); do
                echo "CURRENTLY RUNNING: $model_size" "$dataset" "$attestation" "$i"
                echo "RUN $i:" >> base_results.txt

                # Capture start time in seconds (with nanosecond precision converted to seconds)
                start_time=$(date +%s.%N)

                # Run the Python script and capture its output including start time details
                output=$(python3 main.py --dataset "$dataset" --attestation_type "$attestation" --architecture "$model_size" --exp_id "$i" 2>&1)

                # Capture end time in seconds (with nanosecond precision converted to seconds)
                end_time=$(date +%s.%N)

                # Extract Python start time from the output
                python_start_time=$(echo "$output" | grep "Start Time:" | awk '{print $3}')

                # Compute the whole duration (end time - start time) in seconds using bc
                duration_in_seconds=$(echo "$end_time - $start_time" | bc)

                # Compute the initialization duration (Python start time - Bash start time) in seconds using bc
                init_duration_in_seconds=$(echo "$python_start_time - $start_time" | bc)

                # Log the output from the Python script and the computed durations
                echo "$output" >> base_results.txt
                echo "Initialization duration: $init_duration_in_seconds seconds" >> base_results.txt
                echo "Whole duration: $duration_in_seconds seconds" >> base_results.txt
                echo "-----------------------" >> base_results.txt

            done
        done
    done
done

echo "Test executions completed."
