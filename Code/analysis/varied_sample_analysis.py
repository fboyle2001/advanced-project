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

def plot_final_accuracy_over_size(by_size: Dict[int, Dict[str, TechniqueData]], parent_folder: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = list(by_size.keys())
    
    for technique_name in by_size[xs[0]].keys():
        ys = []

        for size in xs:
            technique = by_size[size][technique_name]

            final_accuracies = np.array([run.tasks[list(run.tasks.keys())[-1]].overall_accuracy for run in technique.runs])

            # Temporary until data point generation completes
            if size == 1000 and parent_folder == "output_cifar100_varied_sorted" and technique_name == "Novel BN":
                final_accuracies = np.array([0.6766, 0.6900])

            ys.append(final_accuracies.mean() * 100)
        
        assert len(ys) == len(xs)
        ax.plot(xs, ys, label=technique_name, marker="o")

    static_techniques = {
        "output_cifar100_varied_sorted": {
            "EWC": np.array([0.0489]),
            "L2P": np.array([0.6240]),
            "Finetuning": np.array([0.0328]),
            "Offline": np.array([0.5804, 0.5742, 0.5891, 0.5894, 0.5755]),
            "ViT Transfer": np.array([0.9167])
        },
        "output_cifar10_varied_sorted": {
            "EWC": np.array([0.1613]),
            "L2P": np.array([0.7560]),
            "Finetuning": np.array([0.1642]),
            "Offline": np.array([0.8995]),
            "ViT Transfer": np.array([0.9895])
        }
    }

    for name, static_accuracy in static_techniques[parent_folder].items():
        ys = (static_accuracy.mean() * 100).repeat(len(xs))
        ax.plot(xs, ys, label=name, marker=None, linestyle="dashed")
    
    ax.grid()
    ax.set_xticks(xs)
    ax.set_ylabel("Final Accuracy (%)")
    ax.set_title("Final Accuracy over Max Memory Size")
    ax.set_ylim(0, 100)
    ax.set_xlim(min(xs), max(xs))
    ax.set_xlabel("Maximum Number of Samples in Memory")
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    return fig
    

def main():
    parent_folder = "output_cifar10_varied_sorted"
    by_size = load_by_buffer_size(f"../{parent_folder}", [200, 500, 1000, 2000, 5000])
    fin_acc_over_size = plot_final_accuracy_over_size(by_size, parent_folder)

    fin_acc_over_size.savefig(f"./variable_sampling/{parent_folder}.png", bbox_inches="tight")

    # plt.show()

if __name__ == "__main__":
    main()