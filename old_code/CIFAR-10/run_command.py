import subprocess
import numpy as np

# Define your command
command = ['gramine-sgx', './io', 'io.py', '--is_sgx', 'y']

# Initialize lists for each timing metric
load_times = []
compute_times = []
form_quote_times = []
initialization_durations = []
whole_durations = []
start_time= []

# Run the command (once, since it internally iterates 100 times)
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Process the output if the command was successful
if result.returncode == 0:
    output_lines = result.stdout.splitlines()
    
    for line in output_lines:
        if "Time to load:" in line:
            load_times.append(float(line.split()[-1]))
        elif "Time to computate:" in line:
            compute_times.append(float(line.split()[-1]))
        elif "Time to form quote:" in line:
            form_quote_times.append(float(line.split()[-1]))
        elif "Duration:" in line:
            whole_durations.append(float(line.split()[-1]))
        elif "Start Time:" in line:
            start_time.append(float(line.split()[-1]))
else:
    print(f"Command failed with error: {result.stderr}")

# Function to calculate mean and std dev
def calculate_stats(times_list):
    if times_list:
        return np.mean(times_list), np.std(times_list)
    else:
        return 0, 0  # Avoid division by zero if the list is empty

# Calculate and save the statistics for each timing metric
# At the end of run_command.py
with open("timing_stats_io.txt", "w") as file:
    # Other metrics
    for metric_name, times in [("Load Time", load_times),
                               ("Compute Time", compute_times),
                               ("Form Quote Time", form_quote_times),
                               ("Whole duration time", whole_durations),]:
        mean_value, std_dev = calculate_stats(times)
        file.write(f"{metric_name} - Mean: {mean_value:.10f}, Standard Deviation: {std_dev:.10f}\n")
    
    # Initialization and Whole Duration
    # Assuming these should be directly listed, not averaged
    for start in start_time:
        file.write(f"Start Time: {start}")
    
