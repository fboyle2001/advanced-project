from typing import Optional, Dict, List
from dataclasses import dataclass

import json
import os

@dataclass
class PerClassData:
    total_samples: int
    true_positives: int
    false_negatives: int
    false_positives: int

    @property
    def accuracy(self):
        return self.true_positives / self.total_samples

@dataclass
class TaskData:
    overall_accuracy: float
    class_results: Dict[str, PerClassData]
    per_task_accuracy: Dict[int, float]

@dataclass
class RunningStats:
    running_duration_seconds: float
    max_vram_bytes: int
    max_ram_bytes: int

    @property
    def max_vram_mb(self) -> float:
        return self.max_vram_bytes / (1024 ** 2)
    
    @property
    def max_ram_mb(self) -> float:
        return self.max_ram_bytes / (1024 ** 2)

@dataclass
class RunData:
    stats: RunningStats
    tasks: Dict[int, TaskData]
    class_ordering: Dict[int, List[str]]

@dataclass
class TechniqueData:
    name: str
    runs: List[RunData]
    
    @staticmethod
    def parse_technique(name: str, folder: str, task_files: Optional[Dict[int, str]] = None):
        # If we don't specify the task to json output mapping use the default
        if task_files is None:
            task_files = {i + 1: f"task_{i}_results.json" for i in [0, 1, 2, 3, 4]}
        
        # Check the folder has valid file structure for the expected output
        dir_run_folders = os.listdir(folder)

        for run_folder in dir_run_folders:
            dir_file_names = os.listdir(f"{folder}/{run_folder}")
            assert "stats.json" in dir_file_names, "Missing stats.json"

            for task_file in task_files.values():
                assert task_file in dir_file_names, f"Missing {task_file}"

        loaded_runs = []

        # Load the data per run
        for sub_dir in dir_run_folders:
            run_folder = f"{folder}/{sub_dir}"

            # Load the data from the stats.json
            stats = None

            with open(f"{run_folder}/stats.json", "r") as stats_fp:
                stats = json.load(stats_fp)

            loaded_stats = RunningStats(
                running_duration_seconds=stats["running_duration"],
                max_vram_bytes=stats["max_gpu_memory_used"],
                max_ram_bytes=stats["max_process_memory_used"]
            )

            # Load the task data
            loaded_tasks = {}
            class_ordering = None

            for task_no, task_file in task_files.items():
                task = None

                with open(f"{run_folder}/{task_file}", "r") as task_fp:
                    task = json.load(task_fp)

                if class_ordering is None:
                    class_ordering = task["task_classes"]
                
                assert class_ordering == task["task_classes"]

                overall_accuracy = task["overall_accuracy"]
                per_class_stats = task["per_class_stats"]
                per_task_accuracy = task["per_task_accuracy"]

                class_results = {}

                for class_name in per_class_stats:
                    class_stats = PerClassData(
                        per_class_stats[class_name]["real_total"],
                        per_class_stats[class_name]["true_positives"],
                        per_class_stats[class_name]["false_negative"],
                        per_class_stats[class_name]["false_positive"]
                    )

                    class_results[class_name] = class_stats
                
                loaded_task = TaskData(
                    overall_accuracy,
                    class_results,
                    {int(t) + 1: per_task_accuracy[t] for t in per_task_accuracy.keys()}
                )

                loaded_tasks[task_no] = loaded_task

            loaded_run = RunData(loaded_stats, loaded_tasks, class_ordering)
            loaded_runs.append(loaded_run)
        
        return TechniqueData(name, loaded_runs)

if __name__ == "__main__":
    data = TechniqueData.parse_technique("test", "../output/der")
    
    print(json.dumps(data, indent=2, default=lambda o: o.__dict__))