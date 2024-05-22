import pulp
import numpy as np

# List of all points in the grid (30x19)
all_points = [(x, y) for x in range(30) for y in range(19)]

# Specific coordinates to cover
specific_points = [(11, 0), (8, 2), (10, 2), (23, 2), (25, 2), (30, 3), (10, 4), (17, 4), (4, 5), (14, 5), (19, 5), (20, 5),
                   (27, 6), (13, 7), (16, 7), (17, 7), (18, 7), (19, 7), (21, 7), (28, 7), (1, 8), (8, 8), (17, 8), (19, 8),
                   (3, 9), (10, 9), (11, 9), (14, 9), (17, 9), (7, 10), (11, 10), (14, 10), (21, 10), (29, 10), (11, 11),
                   (14, 11), (16, 11), (21, 11), (1, 12), (11, 12), (12, 12), (28, 12), (29, 13), (30, 13), (9, 14), (11, 14),
                   (12, 14), (13, 14), (18, 15), (5, 16), (14, 16), (15, 17), (19, 17), (26, 17), (14, 18), (15, 18), (17, 18), (19, 19)]

# Create a distance matrix
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

distance_matrix = {p: {q: euclidean_distance(p, q) for q in specific_points} for p in all_points}

# Set up the optimization problem
model = pulp.LpProblem("Minimal_AED_Placement", pulp.LpMinimize)

# Decision variables: Whether an AED is placed at point p
x = pulp.LpVariable.dicts("x", all_points, cat=pulp.LpBinary)

# Objective function: Minimize the number of AEDs placed
model += pulp.lpSum(x[p] for p in all_points)

# Coverage constraints: Ensure every specific point is covered by at least one AED
for q in specific_points:
    model += pulp.lpSum(x[p] for p in all_points if distance_matrix[p][q] <= 4) >= 1

# Solve the problem
model.solve()

# Output results
print("Status:", pulp.LpStatus[model.status])
print("Minimum number of AEDs needed:", pulp.value(model.objective))
print("Locations to place AEDs:")
for p in all_points:
    if pulp.value(x[p]) == 1:
        print(p)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Assuming `all_points` and `specific_points` are already defined from your optimization setup
# Let's assume the results from your optimization model have provided the AED locations
aed_locations = [(p[0], p[1]) for p in all_points if pulp.value(x[p]) == 1]

# Determine which specific points are covered
covered_points = []
for q in specific_points:
    if any(pulp.value(x[p]) == 1 and distance_matrix[p][q] <= 4 for p in all_points):
        covered_points.append(q)

# Determine uncovered points
uncovered_points = [q for q in specific_points if q not in covered_points]

cols = 19  # Assuming a grid with 19 columns for the y-axis

# Create plot
plt.figure(figsize=(12, 8))
ax = plt.gca()
plt.grid(True)
plt.title('Optimal AED Placement for full coverage')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Function to mirror y-coordinate
mirror_y = lambda y: (cols - 1) - y

# Plotting all points on the grid as a base layer (optional)
for point in all_points:
    plt.plot(point[0], mirror_y(point[1]), 'o', color='lightgray', alpha=0.5)

# Plot specific points
for point in specific_points:
    plt.plot(point[0], mirror_y(point[1]), 'o', color='blue', label='Cardiac arrests' if point == specific_points[0] else "")

# Plot AED locations and their coverage circles
for aed in aed_locations:
    mirrored_aed_y = mirror_y(aed[1])
    plt.plot(aed[0], mirrored_aed_y, 'P', color='gold', markersize=12, label='AED Locations' if aed == aed_locations[0] else "")
    coverage_circle = Circle((aed[0], mirrored_aed_y), radius=4, color='orange', fill=False, linewidth=3, linestyle='dotted', label='within 100m radius')
    ax.add_patch(coverage_circle)

# Add legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # removing duplicate labels
plt.legend(by_label.values(), by_label.keys())

plt.xlim(-1, 30)  # Adjust as necessary
plt.ylim(-1, 19)  # Adjust as necessary
plt.xticks(range(0, 30))
plt.yticks(range(0, 19))
plt.show()

