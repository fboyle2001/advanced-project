import matplotlib.pyplot as plt
import numpy as np

xs = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck", "Avg"]
ys = np.array([941, 955, 853, 803, 886, 820, 929, 929, 943, 936, 899.5]) / 1000 * 100

fig = plt.figure()
ax = fig.add_subplot(111)

bars = ax.bar(x=xs, height=ys, color=["#1f77b4"] * 10 + ["#ff7f0e"])

# for x, bar in zip(xs, bars):
#     print(x, bar.get_x() + bar.get_width() / 2)
#     ax.text(x=bar.get_x() + bar.get_width() / 2, y=bar.get_height() / 2, s="test")

ax.bar_label(bars, [f"{y:.1f}" for y in ys], label_type="center", color="white")
ax.set_xlabel("Classes")
ax.set_ylabel("Accuracy (%)")

plt.show()