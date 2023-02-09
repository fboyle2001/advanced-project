from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

import time
import os
import json

from technique_parser import TechniqueData
from analysis import load_techniques, get_technique_result_structure, NumpyEncoder

def generate_structure(parent: str, buffer_size: int):
    technique_result_structure = {
        "DER": {
            "folder": f"{parent}/{buffer_size}/der",
            "task_files": None
        },
        "DER++": {
            "folder": f"{parent}/{buffer_size}/der_pp",
            "task_files": None
        },
        # "Finetuning": {
        #     "folder": f"{parent}/{buffer_size}/finetuning",
        #     "task_files": None
        # },
        "Rainbow": {
            "folder": f"{parent}/{buffer_size}/rainbow_online",
            "task_files": {i: f"task_{i * 50}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        # "L2P": {
        #     "folder": f"{parent}/{buffer_size}/l2p",
        #     "task_files": None
        # },
        "SCR": {
           "folder": f"{parent}/{buffer_size}/scr",
           "task_files": None 
        },
        "Novel BN": {
            "folder": f"{parent}/{buffer_size}/novel_bn",
            "task_files": {i: f"task_{i * 70}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        # "Novel RD": {
        #     "folder": f"{parent}/{buffer_size}/novel_rd",
        #     "task_files": {i: f"task_{i * 70}_results.json" for i in [1, 2, 3, 4, 5]}
        # },
        # "EWC": {
        #     "folder": f"{parent}/{buffer_size}/ewc",
        #     "task_files": None 
        # },
        "GDumb": {
            "folder": f"{parent}/{buffer_size}/gdumb",
            "task_files": {i: f"task_{256 * i - 6}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        # "Offline": {
        #     "folder": f"{parent}/{buffer_size}/offline",
        #     "task_files": {i: "task_250_results.json" for i in [1, 2, 3, 4, 5]}
        # }
        # "GDumb BD": {
        #     "folder": f"{parent}/{buffer_size}/rainbow_ncm",
        #     "task_files": {i: f"task_{256 * i - 6}_results.json" for i in [1, 2, 3, 4, 5]}
        # }
    }

    technique_result_structure = dict(sorted(technique_result_structure.items()))
    return technique_result_structure

def load_by_buffer_size(parent: str, buffer_sizes: List[int]):
    by_size: Dict[int, Dict[str, TechniqueData]] = {}

    for size in buffer_sizes:
        structure = generate_structure(parent, size)
        techniques = load_techniques(structure)
        by_size[size] = techniques
    
    return by_size

def plot_final_accuracy_over_size(by_size: Dict[int, Dict[str, TechniqueData]]):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = list(by_size.keys())
    
    for technique_name in by_size[xs[0]].keys():
        ys = []

        for size in xs:
            technique = by_size[size][technique_name]
            final_accuracies = np.array([run.tasks[list(run.tasks.keys())[-1]].overall_accuracy for run in technique.runs])
            ys.append(final_accuracies.mean() * 100)
        
        assert len(ys) == len(xs)
        ax.plot(xs, ys, label=technique_name, marker="o")
    
    ax.grid()
    ax.set_xticks(xs)
    ax.set_ylabel("Final Accuracy (%)")
    ax.set_title("Final Accuracy by Max Memory Buffer Sample Count")
    ax.set_ylim(0, 100)
    ax.set_xlim(min(xs), max(xs))
    ax.set_xlabel("Max Memory Buffer Sample Count")
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    return fig
    

def main():
    by_size = load_by_buffer_size("../output_cifar10_varied_sorted", [200, 500, 1000, 5000])
    fin_acc_over_size = plot_final_accuracy_over_size(by_size)

    plt.show()

if __name__ == "__main__":
    main()