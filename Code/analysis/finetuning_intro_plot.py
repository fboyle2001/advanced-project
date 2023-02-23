import matplotlib.pyplot as plt
import numpy as np

xs = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck", "Avg"]

tasks = np.array([
    np.array([0, 0, 0, 78.6, 92.1, 0, 0, 0, 0, 0, 17.1]),
    np.array([0, 92.9, 0, 0, 0, 0, 98.5, 0, 0, 0, 19.1]),
    np.array([92.1, 0, 89.7, 0, 0, 0, 0, 0, 0, 0, 18.2]),
    np.array([0, 0, 0, 0, 0, 94.5, 0, 0, 0, 96.7, 19.1]),
    np.array([0, 0, 0, 0, 0, 0, 0, 98.3, 96.2, 0, 19.4])
])

print(tasks)

for task in range(5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ys = tasks[task]

    bars = ax.bar(x=xs, height=ys, color=["#1f77b4"] * 10 + ["#ff7f0e"])

    ax.bar_label(bars, [f"{y:.1f}" for y in ys], label_type="center", color="white")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

    fig.savefig(f"./finetuning_intro_output/task_{task + 1}.png")