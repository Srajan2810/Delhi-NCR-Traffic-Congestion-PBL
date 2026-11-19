
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.optimize import linprog

# Create folders
os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Decision variables:
# x1 = Green time for North-South
# x2 = Green time for East-West

# Maximize Z = 8*x1 + 5*x2
# linprog performs minimization, so negate coefficients.

c = [-8, -5]

# Total green time constraint
A_eq = [[1, 1]]
b_eq = [120]

# Minimum and maximum green time
bounds = [
    (30, 90),
    (30, 90)
]

# Solve the LP problem
result = linprog(
    c,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=bounds,
    method="highs"
)

print("\n===== LINEAR PROGRAMMING RESULTS =====")

if result.success:

    x1, x2 = result.x

    maximum_value = -result.fun

    print("North-South Green Time:", round(x1, 2), "seconds")
    print("East-West Green Time:", round(x2, 2), "seconds")
    print("Maximum Objective Value:", round(maximum_value, 2))

    print("\nTotal Green Time:", round(x1 + x2, 2))

    # Save results
    with open("results/lp_results.txt", "w") as file:
        file.write("TRAFFIC SIGNAL OPTIMIZATION\n")
        file.write(f"North-South: {x1:.2f} seconds\n")
        file.write(f"East-West: {x2:.2f} seconds\n")
        file.write(f"Objective Value: {maximum_value:.2f}\n")

    # Generate graph
    plt.figure(figsize=(8, 6))

    plt.bar(
        ["North-South", "East-West"],
        [x1, x2]
    )

    plt.title("Optimized Traffic Signal Green Time")
    plt.xlabel("Traffic Signal Phase")
    plt.ylabel("Green Time (seconds)")

    plt.tight_layout()
    plt.savefig("graphs/10_optimized_signal_time.png")
    plt.show()

    print("\nOptimization completed successfully!")

else:
    print("Optimization failed:", result.message)

