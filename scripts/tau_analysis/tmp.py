import numpy as np
import matplotlib.pyplot as plt

# Initialize data again
cost_array = np.random.normal(loc=100, scale=25, size=20)
cost_array_sorted = np.sort(cost_array)[::-1]
average_cost = np.mean(cost_array)

# Plotting the bar plot with specified highlights for the first 3 and last 4 bars, renaming the labels
plt.figure(figsize=(10, 6))
plt.bar(
    range(1, 4), cost_array_sorted[:3], color="lightgreen", label="$k_a$ most loaded"
)
plt.bar(range(4, 21 - 4), cost_array_sorted[3 : 20 - 4], color="skyblue")
plt.bar(
    range(21 - 4, 21),
    cost_array_sorted[-4:],
    color="salmon",
    label="$k_b$ least loaded",
)

# Adding the average line
plt.axhline(
    y=average_cost,
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Average: {average_cost:.2f}",
)

# Updating labels
plt.xlabel("Rank Cost in Sorted Descending Order")
plt.ylabel("Cost")

# No title, show legend and grid
plt.legend()
plt.grid(True)
plt.show()
