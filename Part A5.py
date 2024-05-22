import gurobipy as gp
import math
from gurobipy import GRB
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

# Define grid dimensions and spacing
rows = 9
columns = 10
spacing = 25

# Define candidate AEDs
candidate_AEDs = [(3, 9), (5, 8), (6, 8),(2,6),(3, 6),(4, 6), (5, 6), (7,6), (3, 5), (5, 5), (3, 4), (7, 3), (2, 2), (7, 2)]
candidate_locations = [(3, 9), (5, 8), (6, 8),(2,6),(3, 6),(4, 6), (5, 6), (7,6), (3, 5), (5, 5), (3, 4), (7, 3), (2, 2), (7, 2)]

## Calculate Euclidean distance matrix between candidate_AEDs and candidate_locations
num_AEDs = len(candidate_AEDs)
num_locations = len(candidate_locations)
distance_matrix = np.zeros((num_AEDs, num_locations))

for i in range(num_AEDs):
    for j in range(num_locations):
        x1, y1 = candidate_AEDs[i]
        x2, y2 = candidate_locations[j]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * spacing
        distance_matrix[i, j] = distance

# Round the distance matrix to 1 decimal place
rounded_distance_matrix = np.round(distance_matrix, 0)

# Print the rounded distance matrix
print("Euclidean Distance Matrix:")
print(rounded_distance_matrix)


# Create binary matrix indicating distances less than or equal to 100
a_ij = np.where(distance_matrix <= 100, 1, 0)
print("Coverage Matrix (a_ij):")
print(a_ij)

# Count the number of 1s in each row
count_ones_per_row = np.sum(a_ij, axis=0)

# Print the count of 1s in each row
print("Count of 1s in each row:")
print(count_ones_per_row)

# Keep only rows 1 and 6 (index 0 and 5)
filtered_a_ij = a_ij[[12, 13], :]

# Print the filtered coverage matrix
print("Filtered Coverage Matrix (a_ij) with rows 1 and 6:")
print(filtered_a_ij)

# Keep only rows 1 and 6 (index 0 and 5)
filtered_a_ij = a_ij[[13], :]

# Print the filtered coverage matrix
print("Filtered Coverage Matrix (a_ij) with rows 1 and 6:")
print(filtered_a_ij)


# Create the model
model = gp.Model("AED_Placement")

# Decision variables
y = model.addVars(num_AEDs, vtype=GRB.BINARY, name="y")

# Objective: Minimize the number of AEDs placed
model.setObjective(gp.quicksum(y[i] for i in range(num_AEDs)), GRB.MINIMIZE)

# Coverage constraints: Ensure every cardiac arrest point is covered by at least one AED
for j in range(num_locations):
    model.addConstr(gp.quicksum(a_ij[i, j] * y[i] for i in range(num_AEDs)) >= 1, name=f"cover_{j}")

# Optimize the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print("Number of AEDs located:", model.objVal)
    print("Locations to place AEDs:")
    aed_locations = [candidate_AEDs[i] for i in range(num_AEDs) if y[i].X > 0.5]
    print(aed_locations)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.title('AED Placement and Coverage on a 30x19 Grid')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Plot AED locations
    for aed in aed_locations:
        plt.plot(aed[0], aed[1], 'P', color='gold', markersize=15, label='AED Locations' if aed == aed_locations[0] else "")

    # Plot specific points
    for idx, point in enumerate(candidate_locations):
        if y[idx].X > 0.5:
            plt.plot(point[0], point[1], 'o', color='green', label='Covered Points' if idx == 0 else "")
        else:
            plt.plot(point[0], point[1], 'x', color='red', label='Uncovered Points' if idx == 0 else "")

    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No optimal solution found.")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Retrieve the locations where AEDs are placed
aed_locations = [candidate_AEDs[i] for i in range(len(candidate_AEDs)) if y[i].X > 0.5]

# Determine which points are covered by the AEDs placed
covered_points = [candidate_locations[j] for j in range(len(candidate_locations)) if any(a_ij[i, j] * y[i].X > 0.5 for i in range(len(candidate_AEDs)))]

# Determine which points are not covered
uncovered_points = [q for q in candidate_locations if q not in covered_points]


# Create plot
plt.figure(figsize=(12, 8))
ax = plt.gca()
plt.grid(True)
plt.title('AED Placement and Coverage')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Plotting all points on the grid as a base layer (optional)
for point in candidate_AEDs:
    plt.plot(point[0], point[1], 'o', color='lightgray', alpha=0.5)

# Plot specific points
for point in candidate_locations:
    plt.plot(point[0], point[1], 'o', color='blue', label='Cardiac arrests' if point == candidate_locations[0] else "")

# Plot covered points
for point in covered_points:
    plt.plot(point[0], point[1], '^', color='green', label='Covered cardiac arrests' if point == covered_points[0] else "")

# Plot uncovered points
for point in uncovered_points:
    plt.plot(point[0], point[1], 'x', color='red', label='Uncovered cardiac arrests' if point == uncovered_points[0] else "")

for aed in aed_locations:
    plt.plot(aed[0], aed[1], 'P', color='gold', markersize=12, label='AED Locations' if aed == aed_locations[0] else "")
    coverage_circle = Circle((aed[0], aed[1]), radius=4, color='gold', fill=False, linewidth=3, linestyle='dotted')
    ax.add_patch(coverage_circle)

# Add legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # removing duplicate labels
plt.legend(by_label.values(), by_label.keys())

plt.xlim(-1, 10)  # Adjust as necessary
plt.ylim(-1, 10)  # Adjust as necessary
plt.xticks(range(0, 10))
plt.yticks(range(0, 10))
plt.show()


    