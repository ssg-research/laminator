#!/bin/bash

# Authors: Vasisht Duddu, Oskari Järvinen, Lachlan J Gunn, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Define the number of times you want to run the script
times=10

# Ensure results.txt is empty before starting
> sgx_results.txt

declare -a datasets=("CENSUS" "IMDB" "UTKFACE")
declare -a architectures=("VGG11" "VGG13" "VGG16" "VGG19")
declare -a attestations=("train" "accuracy" "io" "distribution")

echo "Starting tests"

for architecture in "${architectures[@]}"; do
    echo "ARCHITECTURE: $architecture" >> sgx_results.txt

    for dataset in "${datasets[@]}"; do
        echo "DATASET: $dataset" >> sgx_results.txt
        for attestation in "${attestations[@]}"; do
            echo "ATTESTATION: $attestation" >> sgx_results.txt
            if [[ "$dataset" == "IMDB" && "$attestation" == "distribution" ]]; then
                continue
            fi

            # Bind this configuration into the measured enclave: regenerate and
            # re-sign the manifest so DATASET/ATTESTATION_TYPE/ARCHITECTURE are
            # measured into MRENCLAVE for the runs below.
            make SGX=1 DATASET="$dataset" ATTESTATION_TYPE="$attestation" ARCHITECTURE="$architecture"

            for (( i=1; i<=times; i++ )); do
                echo "CURRENTLY RUNNING: $architecture" "$dataset" "$attestation" "$i"
                echo "RUN $i:" >> sgx_results.txt

                # Capture start time in seconds (with nanosecond precision converted to seconds)
                start_time=$(date +%s.%N)

                # Run the Python script and capture its output including start time details
                output=$(EXP_ID="$i" gramine-sgx ./main 2>&1)

                # Capture end time in seconds (with nanosecond precision converted to seconds)
                end_time=$(date +%s.%N)

                # Extract Python start time from the output
                python_start_time=$(echo "$output" | grep "Start Time:" | awk '{print $3}')

                # Compute the whole duration (end time - start time) in seconds using bc
                duration_in_seconds=$(echo "$end_time - $start_time" | bc)

                # Compute the initialization duration (Python start time - Bash start time) in seconds using bc
                init_duration_in_seconds=$(echo "$python_start_time - $start_time" | bc)

                # Log the output from the Python script and the computed durations
                echo "$output" >> sgx_results.txt
                echo "Initialization duration: $init_duration_in_seconds seconds" >> sgx_results.txt
                echo "Whole duration: $duration_in_seconds seconds" >> sgx_results.txt
                echo "-----------------------" >> sgx_results.txt

            done
        done
    done
done

echo "Test executions completed."
