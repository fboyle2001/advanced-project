from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import time
import os
import json

from technique_parser import TechniqueData

# https://stackoverflow.com/a/47626762
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_techniques(technique_structure) -> Dict[str, TechniqueData]:
    techniques = {}

    for name, structure in technique_structure.items():
        loaded_technique = TechniqueData.parse_technique(name, structure["folder"], structure["task_files"])
        techniques[name] = loaded_technique

    return techniques

# Wall clock time over all tasks
def extract_average_wall_clock(techniques: Dict[str, TechniqueData]) -> Dict[str, np.ndarray]:
    results = {}

    for name, technique in techniques.items():
        results[name] = np.array([run.stats.running_duration_seconds for run in technique.runs])
    
    return results

# RAM usage over all tasks
def extract_average_max_ram_usage(techniques: Dict[str, TechniqueData]) -> Dict[str, np.ndarray]:
    results = {}

    for name, technique in techniques.items():
        results[name] = np.array([run.stats.max_ram_mb for run in technique.runs])
    
    return results

# VRAM usage over all tasks
def extract_average_max_vram_usage(techniques: Dict[str, TechniqueData]) -> Dict[str, np.ndarray]:
    results = {}

    for name, technique in techniques.items():
        results[name] = np.array([run.stats.max_vram_mb for run in technique.runs])
    
    return results

# Average accuracy over all tasks averaged
def extract_average_accuracy(techniques: Dict[str, TechniqueData]) -> Dict[str, Dict[int, np.ndarray]]:
    results = {}

    for name, technique in techniques.items():
        per_task = {}

        for run in technique.runs:
            for task_no, task_data in run.tasks.items():
                if task_no not in per_task.keys():
                    per_task[task_no] = np.array([])

                per_task[task_no] = np.append(per_task[task_no], task_data.overall_accuracy)

        results[name] = per_task

    return results

def extract_n_accuracy(techniques: Dict[str, TechniqueData], n: int, top: bool) -> Dict[str, Dict[int, np.ndarray]]:
    results = {}

    for name, technique in techniques.items():
        per_task = {}

        for run in technique.runs:
            for task_no, task_data in run.tasks.items():
                if task_no not in per_task.keys():
                    per_task[task_no] = np.array([])

                accuracy_ordered = sorted(task_data.class_results.values(), key=lambda x: x.accuracy, reverse=top)
                selected = accuracy_ordered[:n]
                averaged_accuracy = np.array([x.accuracy for x in selected]).mean()
                per_task[task_no] = np.append(per_task[task_no], averaged_accuracy)
        
        results[name] = per_task
    
    return results

def plot_wall_clock(techniques: Dict[str, TechniqueData]):
    wall_clock = extract_average_wall_clock(techniques)
    boxplot_data = [data for data in wall_clock.values()]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Technique Wall-Clock Time")
    ax.set_ylabel("Time Taken (s)")

    ax.boxplot(boxplot_data, labels=[k for k in techniques.keys()])
    
    return fig

def plot_memory_usage(techniques: Dict[str, TechniqueData], stacked: bool, bar_width: float = 0.35):
    ram = extract_average_max_ram_usage(techniques)
    vram = extract_average_max_vram_usage(techniques)

    x_labels = [k for k in ram.keys()]
    xs = np.arange(len(x_labels))

    ram_ys = [ram[k].mean() for k in x_labels]
    vram_ys = [vram[k].mean() for k in x_labels]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(xs, x_labels)
    ax.set_title("Peak Memory Usage")
    ax.set_ylabel("Memory (mb)")

    if stacked:
        ax.bar(xs, ram_ys, width=bar_width, label="RAM Usage")
        ax.bar(xs, vram_ys, width=bar_width, label="VRAM Usage")
    else:
        ax.bar(xs - bar_width / 2, ram_ys, width=bar_width, label="RAM Usage")
        ax.bar(xs + bar_width / 2, vram_ys, width=bar_width, label="VRAM Usage")

    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
    
    return fig

def plot_average_accuracy(techniques: Dict[str, TechniqueData]):
    average_accuracy = extract_average_accuracy(techniques)
    tasks = [1, 2, 3, 4, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for name, task_accuracies in average_accuracy.items():
        ys = [accuracies.mean() * 100 for accuracies in task_accuracies.values()]
        ax.plot(tasks, ys, label=name, marker="o")

    ax.grid()
    ax.set_xticks(tasks)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy by Task")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Task")
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    # Error Bars?

    return fig

def plot_n_accuracy(techniques: Dict[str, TechniqueData], n: int, top: bool):
    n_accuracy = extract_n_accuracy(techniques, n=n, top=top)
    tasks = [1, 2, 3, 4, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for name, task_accuracies in n_accuracy.items():
        ys = [accuracies.mean() * 100 for accuracies in task_accuracies.values()]
        ax.plot(tasks, ys, label=name, marker="o")

    ax.grid()
    ax.set_xticks(tasks)
    ax.set_ylabel("Accuracy (%)")
    title = f"{'Top' if top else 'Bottom'}-{n} Accuracy by Task"
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Task")
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    return fig

def main(save: bool, show: bool):
    technique_result_structure = {
        "DER": {
            "folder": "../output/der",
            "task_files": None
        },
        "DER++": {
            "folder": "../output/der_pp",
            "task_files": None
        },
        "Finetuning": {
            "folder": "../output/finetuning",
            "task_files": None
        },
        "Rainbow": {
            "folder": "../output/rainbow_online",
            "task_files": {i: f"task_{i * 50}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        "L2P": {
            "folder": "../output/l2p",
            "task_files": None
        },
        "SCR": {
           "folder": "../output/scr",
           "task_files": None 
        },
        "Novel BN": {
            "folder": "../output/novel_bn",
            "task_files": {i: f"task_{i * 70}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        "Novel RD": {
            "folder": "../output/novel_rd",
            "task_files": {i: f"task_{i * 70}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        "EWC": {
            "folder": "../output/ewc",
            "task_files": None 
        },
        "GDumb": {
            "folder": "../output/gdumb",
            "task_files": {i: f"task_{256 * i - 6}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        # "GDumb BD": {
        #     "folder": "../output/rainbow_ncm",
        #     "task_files": {i: f"task_{256 * i - 6}_results.json" for i in [1, 2, 3, 4, 5]}
        # }
    }

    technique_result_structure = dict(sorted(technique_result_structure.items()))

    store_dir = f"./output/{time.time()}"
    os.makedirs(store_dir, exist_ok=False)

    techniques = load_techniques(technique_result_structure)

    top_5_fig = plot_n_accuracy(techniques, n=5, top=True)
    bottom_5_fig = plot_n_accuracy(techniques, n=5, top=False)
    avg_accuracy_fig = plot_average_accuracy(techniques)
    memory_stacked_fig = plot_memory_usage(techniques, stacked=True)
    memory_grouped_fig = plot_memory_usage(techniques, stacked=False)
    wc_fig = plot_wall_clock(techniques)

    top_5 = extract_n_accuracy(techniques, n=5, top=True)
    bottom_5 = extract_n_accuracy(techniques, n=5, top=False)
    avg_acc = extract_average_accuracy(techniques)
    ram = extract_average_max_ram_usage(techniques)
    vram = extract_average_max_vram_usage(techniques)
    wc = extract_average_wall_clock(techniques)

    joined = {}

    for technique in top_5.keys():
        joined[technique] = {
            "top_5": top_5[technique],
            "bottom_5": bottom_5[technique],
            "avg_acc": avg_acc[technique],
            "ram": ram[technique],
            "vram": vram[technique],
            "wc": wc[technique]
        }

    if save:
        top_5_fig.savefig(f"{store_dir}/Top 5 Accuracy.png", bbox_inches="tight")
        bottom_5_fig.savefig(f"{store_dir}/Bottom 5 Accuracy.png", bbox_inches="tight")
        avg_accuracy_fig.savefig(f"{store_dir}/Average Accuracy.png", bbox_inches="tight")
        memory_stacked_fig.savefig(f"{store_dir}/Stacked Memory Usage.png", bbox_inches="tight")
        memory_grouped_fig.savefig(f"{store_dir}/Grouped Memory Usage.png", bbox_inches="tight")
        wc_fig.savefig(f"{store_dir}/Wall Clock Time.png", bbox_inches="tight")

        with open(f"{store_dir}/processed_results.json", "w+") as fp:
            json.dump(joined, fp, indent=2, cls=NumpyEncoder)

    if show:
        plt.show()

    """
    Plots:
    * Wall clock time with error bars -> box plot
    * Memory usage -> stacked bar chart
    * Average accuracy -> multi-line graph
    * Top and bottom 5 -> needs to be done on the final output only?
    """

if __name__ == "__main__":
    main(save=True, show=False)