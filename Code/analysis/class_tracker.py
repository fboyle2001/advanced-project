from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import time
import os
import json

from technique_parser import TechniqueData
from analysis import COLOUR_MAP, load_techniques, get_technique_result_structure, NumpyEncoder

def track_class_accuracy(techniques: Dict[str, TechniqueData], class_name: str):
    results = {}

    for name, technique in techniques.items():
        per_task = {}

        for run in technique.runs[:1]:
            for task_no, task_data in run.tasks.items():
                class_accuracy = task_data.class_results[class_name].accuracy

                if task_no not in per_task.keys():
                    per_task[task_no] = np.array([])

                per_task[task_no] = np.append(per_task[task_no], class_accuracy)
        
        results[name] = per_task
     
    return results

def track_first_task(techniques: Dict[str, TechniqueData]):
    results = {}
    exp_first_classes = {}

    for name, technique in techniques.items():
        per_task = {}

        for run_id, run in enumerate(technique.runs):
            first_classes = run.class_ordering["0"]

            if run_id not in exp_first_classes:
                exp_first_classes[run_id] = first_classes
            
            assert exp_first_classes[run_id] == first_classes, "Ordering error"

            for task_no, task_data in run.tasks.items():
                class_accuracies = []

                for class_name in first_classes:
                    class_accuracy = task_data.class_results[class_name].accuracy
                    class_accuracies.append(class_accuracy)

                if task_no not in per_task.keys():
                    per_task[task_no] = np.array([])

                per_task[task_no] = np.append(per_task[task_no], np.array(class_accuracies).mean())
        
        results[name] = per_task

    return results

def track_first_class(techniques: Dict[str, TechniqueData]):
    results = {}
    exp_first_classes = {}

    for name, technique in techniques.items():
        per_task = {}

        for run_id, run in enumerate(technique.runs):
            first_class = run.class_ordering["0"][0]

            if run_id not in exp_first_classes:
                exp_first_classes[run_id] = first_class
            
            assert exp_first_classes[run_id] == first_class, "Ordering error"

            for task_no, task_data in run.tasks.items():
                class_accuracy = task_data.class_results[first_class].accuracy

                if task_no not in per_task.keys():
                    per_task[task_no] = np.array([])

                per_task[task_no] = np.append(per_task[task_no], class_accuracy)
        
        results[name] = per_task

    return results

def plot_accuracy_over_task(data):
    tasks = np.array([1, 2, 3, 4, 5])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_prop_cycle("color", COLOUR_MAP)

    section_width = 0.8
    num_techniques = len(data.keys())
    width = section_width / num_techniques

    for i, (name, task_accuracies) in enumerate(data.items()):
        task_means = [per_task.mean() * 100 for per_task in task_accuracies.values()]        
        print(name, task_means)
        pos = tasks - (section_width / 2) + (i / (num_techniques - 1)) * section_width
        ax.bar(pos, task_means, width=width, label=name)

    ax.vlines([1.5, 2.5, 3.5, 4.5], 0, 100, linestyles="dashed", colors="gray")

    ax.set_xlim(1 - (section_width / 2) - 0.1, 5 + (section_width / 2) + 0.1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Task")
    ax.set_ylabel("First Task Accuracy (%)")

    # ax.set_title("CIFAR-100 5K Samples First Task Accuracy")
    
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
    return fig


def main():
    folder = "output_cifar10_0.5k"
    out_name = "_".join(folder.split("_")[1:]) + ".png"
    technique_result_structure = get_technique_result_structure(folder)
    techniques = load_techniques(technique_result_structure)

    first_class_only = track_first_class(techniques)
    first_task = track_first_task(techniques)
    fig = plot_accuracy_over_task(first_task)
    fig.savefig(f"./first_task_forgetting/{out_name}", bbox_inches="tight")

    # plot the first class averages

    # print(json.dumps(tracked, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()