# Learning From Data Problem Set 4.4 - 7
# Bias-Variance Tradeoff Analysis

import numpy as np
import matplotlib.pyplot as plt


def compute_best_fit_slope_through_origin(x1, y1, x2, y2):
    return (x1 * y1 + x2 * y2) / (x1**2 + x2**2)

TRIALS = 1000000

slopes_through_origin_large = []

for _ in range(TRIALS):
    x1, x2 = np.random.uniform(-1, 1, 2)
    y1, y2 = np.sin(np.pi * x1), np.sin(np.pi * x2)
    a = compute_best_fit_slope_through_origin(x1, y1, x2, y2)
    slopes_through_origin_large.append(a)

average_slope_through_origin_large = np.mean(slopes_through_origin_large)
average_slope_through_origin_large


## Graphing ##

def target_function(x):
    return np.sin(np.pi * x)

x_values = np.linspace(-1, 1, 400)

plt.figure(figsize=(10, 6))
plt.plot(x_values, target_function(x_values), label="Target Function: $f(x) = \sin(\pi x)$", color="black", linewidth=2)

for a in slopes_through_origin_large[:100]:
    plt.plot(x_values, a * x_values, color="lightblue", linewidth=0.5)

plt.plot(x_values, average_slope_through_origin_large * x_values, label="Average Hypothesis: $\overline{g}(x) = 1.43x$", color="red", linewidth=2)

plt.ylim([-1, 1])
plt.xticks(np.arange(-1, 1.1, 0.5))
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.legend()
plt.title("Target Function, Hypotheses, and Average Hypothesis with Restricted Y-axis")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
