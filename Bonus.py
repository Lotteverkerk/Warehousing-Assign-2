import pulp
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Grid dimensions
rows = 30
cols = 19

# Create all grid points
points = [(x, y) for x in range(rows) for y in range(cols)]

# Specific coordinates to cover
specific_points = [(11, 0), (8, 2), (10, 2), (23, 2), (25, 2), (30, 3), (10, 4), (17, 4), (4, 5), (14, 5), (19, 5), (20, 5),
                   (27, 6), (13, 7), (16, 7), (17, 7), (18, 7), (19, 7), (21, 7), (28, 7), (1, 8), (8, 8), (17, 8), (19, 8),
                   (3, 9), (10, 9), (11, 9), (14, 9), (17, 9), (7, 10), (11, 10), (14, 10), (21, 10), (29, 10), (11, 11),
                   (14, 11), (16, 11), (21, 11), (1, 12), (11, 12), (12, 12), (28, 12), (29, 13), (30, 13), (9, 14), (11, 14),
                   (12, 14), (13, 14), (18, 15), (5, 16), (14, 16), (15, 17), (19, 17), (26, 17), (14, 18), (15, 18), (17, 18), (19, 19)]

# Euclidean distance function
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Probability of saving a patient based on distance
def probability_of_saving(distance):
    if distance <= 2:
        return 1
    elif distance <= 4:
        return 0.5
    return 0

# Set up the optimization problem
model = pulp.LpProblem("AED_Placement", pulp.LpMaximize)

# Decision variable: Whether an AED is placed at point p
x = pulp.LpVariable.dicts("x", points, cat=pulp.LpBinary)

# Calculate expected lives saved for each point
expected_savings = {p: sum(probability_of_saving(euclidean_distance(p, q)) for q in specific_points) for p in points}

# Objective function: Maximize the expected number of lives saved
model += pulp.lpSum(x[p] * expected_savings[p] for p in points)

# Constraint: Only 1 AED can be placed
model += pulp.lpSum(x[p] for p in points) == 1

# Solve the problem
model.solve()

# Find the best location and expected number of lives saved
best_location = None
max_lives_saved = 0
for p in points:
    if pulp.value(x[p]) == 1:
        best_location = p
        max_lives_saved = expected_savings[p]
        break

lives_saved_prob_1 = 0
lives_saved_prob_05 = 0

if best_location:
    for point in specific_points:
        dist = euclidean_distance(best_location, point)
        if dist <= 2:
            lives_saved_prob_1 += 1
        elif dist <= 4:
            lives_saved_prob_05 += 1

print("Best AED Location:", best_location)
print("Maximum Expected Lives Saved:", max_lives_saved)
print("Number of lives saved with probability 1:", lives_saved_prob_1)
print("Number of lives saved with probability 0.5:", lives_saved_prob_05)

# Plot results
plt.figure(figsize=(12, 8))
ax = plt.gca()
plt.grid(True)
plt.title('AED Placement and survival probability radius')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Calculate the mirrored y-coordinates
mirror_y = lambda y: (cols - 1) - y  # cols is 19, so cols-1 is 18

# Plot all points with inverted y-values
for point in points:
    plt.plot(point[0], mirror_y(point[1]), 'o', color='lightgray', alpha=0.5)

# Plot specific points with inverted y-values
for point in specific_points:
    plt.plot(point[0], mirror_y(point[1]), 'o', color='blue', label='Cardiac arrests' if point == specific_points[0] else "")

# Highlight best location and its coverage, adjusting for the mirrored y-coordinate
if best_location:
    mirrored_best_y = mirror_y(best_location[1])
    plt.plot(best_location[0], mirrored_best_y, 'P', color='gold', markersize=15, label='AED Location')
    # Draw coverage circles for probability thresholds at the mirrored location
    circle1 = Circle((best_location[0], mirrored_best_y), 2, color='green', fill=False, linewidth=2, linestyle='solid', label='100% Coverage')
    circle2 = Circle((best_location[0], mirrored_best_y), 4, color='orange', fill=False, linewidth=2, linestyle='dashed', label='50% Coverage')
    ax.add_patch(circle1)
    ax.add_patch(circle2)

plt.legend()
plt.xlim(-1, 30)
plt.ylim(-1, 19)  # Set limits to original because we're transforming the data, not the axes
plt.xticks(range(0, 30))
plt.yticks(range(0, 19))
plt.show()