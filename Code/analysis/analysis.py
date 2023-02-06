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

def extract_average_forgetting(techniques: Dict[str, TechniqueData]) -> Dict[str, Dict[int, np.ndarray]]:
    results = {}

    for name, technique in techniques.items():
        per_task = {}

        for run in technique.runs:
            # 1. Compute accuracy for tasks 1 to i - 1 at task i
            # accuracy_matrix[i, j] = accuracy on task j < i after learning task m
            accuracy_matrix = np.zeros((5, 5))

            for task_no, task_data in run.tasks.items():
                i = task_no - 1

                for j in range(task_no):
                    a_i_j = task_data.per_task_accuracy[j + 1]
                    accuracy_matrix[i, j] = a_i_j

            # if name.lower() == "ewc":
            #     print(accuracy_matrix)

            # 2. Compute forgetting matrix
            # forgetting_matrix[i, j] = max(a_t_j - a_i_j) for t = 1, ..., i - 1
            forgetting_matrix = np.zeros((5, 5))

            for i in range(1, 5):
                for j in range(i):
                    #m_i = max([accuracy_matrix[t, j] for t in range(0, i)]) + 1e-6
                    f_i_j  = max([accuracy_matrix[t, j] - accuracy_matrix[i, j] for t in range(0, i)])
                    forgetting_matrix[i, j] = f_i_j # / (i * 0.2) #/ m_i

            # if name.lower() == "ewc":
            #     print(forgetting_matrix)
            #     print((forgetting_matrix.sum(axis=0)[:-1] / np.array([4, 3, 2, 1])) / accuracy_matrix.diagonal()[-1])
            #     print(accuracy_matrix.diagonal()[-1])
            
            # 3. Compute average forgetting for each task

            F_ks = []

            for k in range(1, 5):
                if k + 1 not in per_task.keys():
                    per_task[k + 1] = np.array([])

                F_k = (1 / k) * sum([forgetting_matrix[k, i] for i in range(k)])
                per_task[k + 1] = np.append(per_task[k + 1], F_k)
                F_ks.append(F_k)
            
            # if name.lower() == "ewc":
            #     print(F_ks)
            #     print()
                
        results[name] = per_task

    return results

def plot_wall_clock(techniques: Dict[str, TechniqueData]):
    wall_clock = extract_average_wall_clock(techniques)

    print("Wall Clock")

    for name, task in wall_clock.items():
        mean = task.mean()
        se = task.std(ddof=1) / np.sqrt(task.shape[0])
        
        print(name, f"{mean:.2f} +- {se:.2f}")
    
    print()

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

    print("RAM Usage")

    for name, task in ram.items():
        mean = task.mean()
        se = task.std(ddof=1) / np.sqrt(task.shape[0])
        
        print(name, f"{mean:.2f} +- {se:.2f}")
    
    print()

    print("VRAM Usage")

    for name, task in vram.items():
        mean = task.mean()
        se = task.std(ddof=1) / np.sqrt(task.shape[0])
        
        print(name, f"{mean:.2f} +- {se:.2f}")
    
    print()

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

def plot_average_accuracy(techniques: Dict[str, TechniqueData], static_name: str):
    average_accuracy = extract_average_accuracy(techniques)
    tasks = [1, 2, 3, 4, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    print("Average Accuracy")

    for name, task_accuracies in average_accuracy.items():
        ys = [accuracies.mean() * 100 for accuracies in task_accuracies.values()]
        
        overall = list(task_accuracies.values())[-1]
        mean = overall.mean() * 100
        se = overall.std(ddof=1) / np.sqrt(overall.shape[0])
        
        print(name, f"{mean:.2f} +- {se:.2f}")
        ax.plot(tasks, ys, label=name, marker="o")

    print()

    static_techniques = {
        "output_cifar100_5k": {
            "Offline": np.array([0.5804, 0.5742, 0.5891, 0.5894, 0.5755]),
            "ViT Transfer": np.array([0.9167])
        },
        "output_cifar10_5k": {
            "Offline": np.array([0.8995]),
            "ViT Transfer": np.array([0.9895])
        },
        "output_cifar10_0.5k": {
            "Offline": np.array([0.8995]),
            "ViT Transfer": np.array([0.9895])
        }
    }

    for name, static_accuracy in static_techniques[static_name].items():
        ys = (static_accuracy.mean() * 100).repeat(len(tasks))
        ax.plot(tasks, ys, label=name, marker=None, linestyle="dashed")

    ax.grid()
    ax.set_xticks(tasks)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy by Task")
    ax.set_ylim(0, 100)
    ax.set_xlim(min(tasks), max(tasks))
    ax.set_xlabel("Task")
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    # Error Bars?

    return fig

def plot_n_accuracy(techniques: Dict[str, TechniqueData], n: int, top: bool):
    n_accuracy = extract_n_accuracy(techniques, n=n, top=top)
    tasks = [1, 2, 3, 4, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = f"{'Top' if top else 'Bottom'}-{n} Accuracy by Task"

    print(title)
    lowest = 100
    highest = 0

    for name, task_accuracies in n_accuracy.items():
        ys = [accuracies.mean() * 100 for accuracies in task_accuracies.values()]

        if min(ys) < lowest:
            lowest = min(ys)

        if max(ys) > highest:
            highest = max(ys)
        
        overall = list(task_accuracies.values())[-1]
        mean = overall.mean() * 100
        se = overall.std(ddof=1) / np.sqrt(overall.shape[0])    

        print(name, f"{mean:.2f} +- {se:.2f}")
        ax.plot(tasks, ys, label=name, marker="o")

    print()

    y_low = np.floor(lowest / 20) * 20
    y_high = np.ceil(highest / 20) * 20

    ax.grid()
    ax.set_xticks(tasks)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("Task")
    ax.set_xlim(min(tasks), max(tasks))
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    return fig

def plot_average_forgetting(techniques: Dict[str, TechniqueData]):
    average_forgetting = extract_average_forgetting(techniques)
    tasks = [2, 3, 4, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    print("Average Forgetting")

    for name, task_forgetting in average_forgetting.items():
        ys = [forgetting.mean() * 100 for forgetting in task_forgetting.values()]
        
        overall = list(task_forgetting.values())[-1]
        mean = overall.mean() * 100
        se = overall.std(ddof=1) / np.sqrt(overall.shape[0])
        
        print(name, f"{mean:.2f} +- {se:.2f}")
        ax.plot(tasks, ys, label=name, marker="o")

    print()

    ax.grid()
    ax.set_xticks(tasks)
    ax.set_ylabel("Average Forgetting (%)")
    ax.set_title("Average Forgetting per Task")
    ax.set_ylim(0, 40)
    ax.set_xlim(min(tasks), max(tasks))
    ax.set_xlabel("Task")
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    # Error Bars?

    return fig

def get_technique_result_structure(folder: str):
    technique_result_structure = {
        "DER": {
            "folder": f"../{folder}/der",
            "task_files": None
        },
        "DER++": {
            "folder": f"../{folder}/der_pp",
            "task_files": None
        },
        "Finetuning": {
            "folder": f"../{folder}/finetuning",
            "task_files": None
        },
        "Rainbow": {
            "folder": f"../{folder}/rainbow_online",
            "task_files": {i: f"task_{i * 50}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        "L2P": {
            "folder": f"../{folder}/l2p",
            "task_files": None
        },
        "SCR": {
           "folder": f"../{folder}/scr",
           "task_files": None 
        },
        "Novel BN": {
            "folder": f"../{folder}/novel_bn",
            "task_files": {i: f"task_{i * 70}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        "Novel RD": {
            "folder": f"../{folder}/novel_rd",
            "task_files": {i: f"task_{i * 70}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        "EWC": {
            "folder": f"../{folder}/ewc",
            "task_files": None 
        },
        "GDumb": {
            "folder": f"../{folder}/gdumb",
            "task_files": {i: f"task_{256 * i - 6}_results.json" for i in [1, 2, 3, 4, 5]}
        },
        # "Offline": {
        #     "folder": f"../{folder}/offline",
        #     "task_files": {i: "task_250_results.json" for i in [1, 2, 3, 4, 5]}
        # }
        # "GDumb BD": {
        #     "folder": f"../{folder}/rainbow_ncm",
        #     "task_files": {i: f"task_{256 * i - 6}_results.json" for i in [1, 2, 3, 4, 5]}
        # }
    }

    technique_result_structure = dict(sorted(technique_result_structure.items()))
    return technique_result_structure

def main(save: bool, show: bool):
    folder = "output_cifar10_0.5k"
    technique_result_structure = get_technique_result_structure(folder)

    store_dir = f"./output/{time.time()}"
    os.makedirs(store_dir, exist_ok=False)

    techniques = load_techniques(technique_result_structure)

    n_size = 1

    top_n_fig = plot_n_accuracy(techniques, n=n_size, top=True)
    bottom_n_fig = plot_n_accuracy(techniques, n=n_size, top=False)
    avg_accuracy_fig = plot_average_accuracy(techniques, static_name=folder)
    memory_stacked_fig = plot_memory_usage(techniques, stacked=True)
    memory_grouped_fig = plot_memory_usage(techniques, stacked=False)
    wc_fig = plot_wall_clock(techniques)
    avg_forgetting_fig = plot_average_forgetting(techniques)

    top_n = extract_n_accuracy(techniques, n=n_size, top=True)
    bottom_n = extract_n_accuracy(techniques, n=n_size, top=False)
    avg_acc = extract_average_accuracy(techniques)
    ram = extract_average_max_ram_usage(techniques)
    vram = extract_average_max_vram_usage(techniques)
    wc = extract_average_wall_clock(techniques)
    avg_forgetting = extract_average_forgetting(techniques)

    joined = {}

    for technique in top_n.keys():
        joined[technique] = {
            f"top_{n_size}": top_n[technique],
            f"bottom_{n_size}": bottom_n[technique],
            "avg_acc": avg_acc[technique],
            "ram": ram[technique],
            "vram": vram[technique],
            "wc": wc[technique],
            "avg_forgetting": avg_forgetting[technique]
        }

    if save:
        top_n_fig.savefig(f"{store_dir}/Top {n_size} Accuracy.png", bbox_inches="tight")
        bottom_n_fig.savefig(f"{store_dir}/Bottom {n_size} Accuracy.png", bbox_inches="tight")
        avg_accuracy_fig.savefig(f"{store_dir}/Average Accuracy.png", bbox_inches="tight")
        memory_stacked_fig.savefig(f"{store_dir}/Stacked Memory Usage.png", bbox_inches="tight")
        memory_grouped_fig.savefig(f"{store_dir}/Grouped Memory Usage.png", bbox_inches="tight")
        wc_fig.savefig(f"{store_dir}/Wall Clock Time.png", bbox_inches="tight")
        avg_forgetting_fig.savefig(f"{store_dir}/Average Forgetting.png", bbox_inches="tight")

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