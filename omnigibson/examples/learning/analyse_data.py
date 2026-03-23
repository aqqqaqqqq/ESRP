import os
import re
from datetime import datetime
import ast
from tqdm import tqdm
import json

import numpy as np
import matplotlib.pyplot as plt

def exponential_moving_average(data, alpha=0.0005):
    """
    Compute the Exponential Moving Average (EMA) for a given data series.
    
    Parameters:
    - data (list of float): The list of data points to smooth.
    - alpha (float): The smoothing factor, between 0 and 1. Higher alpha gives more weight to recent data.
    
    Returns:
    - ema (list of float): The smoothed data using EMA.
    """
    ema = []
    current_ema = data[0]  # Start with the first data point
    ema.append(current_ema)

    for point in data[1:]:
        current_ema = alpha * point + (1 - alpha) * current_ema
        ema.append(current_ema)

    return ema

def visualize_grasping_objs_mean(entries, alpha=0.05):
    """
    Given a list of entries, calculate the mean of 'grasping_objs' for each entry,
    apply exponential moving average (EMA) for smoothing, and plot the results.

    Parameters:
    - entries (list of dict): List of entries containing 'grasping_objs' and 'timestamp_str'
    - alpha (float): Smoothing factor for EMA (default is 0.1)
    
    Returns:
    - None: Generates a plot of smoothed mean values over time.
    """
    # Initialize lists for timestamps and mean values
    timestamps = []
    mean_values = []
    
    # Loop through each entry in the entries list
    for entry in entries:
        grasping_objs = entry.get('grasping_objs', [])
        rewards = entry.get('rewards', [])
        
        if grasping_objs:  # Check if grasping_objs is not empty
            mean_value = np.sum(rewards) - 0.01 * np.sum(grasping_objs) # Calculate mean of grasping_objs
            timestamps.append(entry['timestamp_str'])
            mean_values.append(mean_value)
    
    # Apply Exponential Moving Average (EMA) to smooth the mean values
    smoothed_values = exponential_moving_average(mean_values, alpha)
    
    # Plotting the smoothed mean grasping objects over time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, mean_values, marker='o', linestyle='-', color='b', label='Mean Grasping Objects', alpha=0.5)
    plt.plot(timestamps, smoothed_values, marker='x', linestyle='-', color='r', label='Smoothed (EMA)')

    # Formatting the plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Timestamp')
    plt.ylabel('Mean of Grasping Objects')
    plt.ylim([-0.5, 2])
    plt.title('Mean of Grasping Objects Over Time (Smoothed with EMA)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()



def extract_and_filter_logs(folder_path):
    log_entries = []  # List to store valid log entries
    zero_entries = []  # List to store entries with all zeros in grasping_objs
    timestamp_counts = {}  # Dictionary to count duplicate timestamps
    
    # Define a pattern to detect the first line of a valid log entry
    entry_start_pattern = re.compile(r"(\d+-\d+-\d+ \d+:\d+:\d+\.\d+)\s+\| INFO\s+\| __main__:main:+\d+ - (.*)$")

    with open("/home/user/Desktop/rearrange/OmniGibson-Rearrange/test_all_data.txt", 'r') as f:
        content = f.read()  # Entire file content as one string
    
    all_num = [0, 0, 0]
    success_num = [0, 0, 0]
    each_arrival = [[],[],[]]
    each_potential = [[],[],[]]
    steps = [0,0,0]
    filename = "ppo_lstm_all_18000_0.3_3.log"
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r") as f:
        for line in f:
            match = entry_start_pattern.search(line)
            if match:
                # print(all_num)
                time_step = match.group(1)
                dict_str = match.group(2)
                dict_str = dict_str.replace("'", '"')
                dict_str = re.sub(r'tensor\(([\deE\+\-\.]+)\)', r'\1', dict_str)
                dict_str = dict_str.replace("False", "false").replace("True", "true")
                data = json.loads(dict_str)
                if data["scene_name"] in content:
                    if data["success"]:
                        if data["obj_num"] < 2:
                            success_num[0] += 1
                            # print(data["scene_name"], data["obj_num"], data["arrival_num"])
                            # steps[0] += data["step"]
                        elif data["obj_num"] > 3:
                            success_num[2] += 1
                            # steps[2] += data["step"]
                        else:
                            success_num[1] += 1
                            # steps[1] += data["step"]
                    if data["obj_num"] < 2:
                        all_num[0] += 1
                        each_arrival[0].append(data["arrival_num"] / data['obj_num']) 
                        each_potential[0].append(data["fini_potential"] / data["init_potential"])
                    elif data["obj_num"] > 3:
                        all_num[2] += 1
                        each_arrival[2].append(data["arrival_num"] / data['obj_num']) 
                        each_potential[2].append(data["fini_potential"] / data["init_potential"])
                    else:
                        all_num[1] += 1
                        each_arrival[1].append(data["arrival_num"] / data['obj_num']) 
                        each_potential[1].append(data["fini_potential"] / data["init_potential"])
                    # each_arrival.append(data["arrival_num"] / data['obj_num']) 
                    # each_potential.append(data["fini_potential"] / data["init_potential"])
                    # all_num += 1
    print(all_num)
    # return success_num / all_num, sum(each_arrival)/all_num, sum(each_potential) / all_num
    return success_num, all_num, each_arrival, each_potential, steps


# Example usage:
folder_path = "/home/user/Desktop/rearrange_logs"  # Your folder path
success_num, all_num, each_arrival, each_potential, steps = extract_and_filter_logs(folder_path)
print("1 object: ", "SR:", success_num[0]/all_num[0], "OSR:", sum(each_arrival[0])/all_num[0], "RDR:", sum(each_potential[0])/all_num[0])
print("2-3 objects: ", "SR:", success_num[1]/all_num[1], "OSR:", sum(each_arrival[1])/all_num[1], "RDR:", sum(each_potential[1])/all_num[1],)
print("4-6 objects: ", "SR:", success_num[2]/all_num[2], "OSR:", sum(each_arrival[2])/all_num[2], "RDR:", sum(each_potential[2])/all_num[2],)
success_sum = sum(success_num)
all_sum = sum(all_num)
arrival_sum = sum(each_arrival[0]) + sum(each_arrival[1]) + sum(each_arrival[2])
potential_sum = sum(each_potential[0]) + sum(each_potential[1]) + sum(each_potential[2])
print("all: ", "SR:", success_sum/all_sum, "OSR:", arrival_sum/all_sum, "RDR:", potential_sum/all_sum)

"""
random:
1 object:  SR: 0.12394366197183099 OSR: 0.12112676056338029 RDR: 1.015212389832849
2-3 objects:  SR: 0.005434782608695652 OSR: 0.012681159420289854 RDR: 1.059485552370269
4-6 objects:  SR: 0.0 OSR: 0.0 RDR: 0.9928549161568647

model:
1 object:  SR: 0.2704225352112676 OSR: 0.2676056338028169 RDR: 0.91627281742207
2-3 objects:  SR: 0.0 OSR: 0.002717391304347826 RDR: 1.060284398019572
4-6 objects:  SR: 0.0 OSR: 0.0 RDR: 1.0425456902670551
"""
