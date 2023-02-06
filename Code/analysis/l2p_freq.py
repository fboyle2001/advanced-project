from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

import os
import re

def load_logs(parent_folder: str):
    logs = {}

    for folder in os.listdir(parent_folder):
        with open(f"{parent_folder}/{folder}/log.log") as fp:
            data = fp.readlines()
            logs[folder] = data
    
    return logs

def extract_frequency_per_task(log: List[str]):
    current_task = 0
    last_seen_freq = np.array([])

    tasks = {}

    for i, line in enumerate(log):
        if "algorithms.l2p:train:163" in line:
            tasks[current_task] = last_seen_freq
            current_task += 1
            last_seen_freq = np.array([])
        
        if "algorithms.l2p:train:263" in line:
            next_line = log[i + 1]
            combined = line.strip() + next_line.strip()
            combined = combined.replace(".", "")
            extracted_tensor = re.findall(r'\[.*?\]', combined)[0].replace("[", "").replace("]", "").split(",")
            freqs = []

            for part in extracted_tensor:
                freqs.append(int(part.strip()))
            
            freqs = np.array(freqs)
            last_seen_freq = freqs
    
    tasks[current_task] = last_seen_freq
    del tasks[0]
    return tasks

def plot_frequency_per_task_style1(frequencies: Dict[int, np.ndarray]):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    section_width = 0.8
    num_parts = 10
    width = section_width / num_parts

    for task, parts in frequencies.items():
        pos = np.array([i / 11 for i in range(10)]) - (section_width / 2) + task
        ax.bar(pos, parts, width=width)
        
    ax.set_xlabel("Task")
    ax.set_ylabel("Prompt Frequency")
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
    return fig

def plot_frequency_per_task_style2(frequencies: Dict[int, np.ndarray]):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    prompts = {}
    tasks = np.array([1, 2, 3, 4, 5])
    section_width = 0.8
    num = 10
    width = section_width / num
    maximum = 0

    for task in frequencies.values():
        for i, part in enumerate(task):
            if i not in prompts.keys():
                prompts[i] = np.array([])
            
            prompts[i] = np.append(prompts[i], part)
            maximum = maximum if part < maximum else part
        
    for prompt_no, freqs in prompts.items():
        pos = tasks - (section_width / 2) + (prompt_no / (num - 1)) * section_width
        ax.bar(pos, freqs, width=width, label=f"Prompt {prompt_no + 1}")

    ax.vlines([1.5, 2.5, 3.5, 4.5], 0, maximum + 200, linestyles="dashed", colors="gray")
    ax.set_ylim(0, maximum + 200)
    ax.set_xlim(1 - (section_width / 2) - 0.1, 5 + (section_width / 2) + 0.1)
    ax.set_xlabel("Task")
    ax.set_ylabel("Prompt Frequency")
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    return fig

def main():
    loaded_logs = load_logs("../output_cifar100_5k/l2p")
    frequencies = extract_frequency_per_task(loaded_logs[list(loaded_logs.keys())[0]])

    # fig = plot_frequency_per_task_style1(frequencies)
    fig2 = plot_frequency_per_task_style2(frequencies)

    fig2.savefig("l.png", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()