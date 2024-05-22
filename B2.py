import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# Define the grid dimensions
rows, cols = 30, 19
all_points = [(x, y) for x in range(rows) for y in range(cols)]

# Define specific coordinates to be covered
specific_points = [(11, 0), (8, 2), (10, 2), (23, 2), (25, 2), (30, 3), (10, 4), (17, 4), (4, 5), (14, 5), (19, 5), (20, 5),
                   (27, 6), (13, 7), (16, 7), (17, 7), (18, 7), (19, 7), (21, 7), (28, 7), (1, 8), (8, 8), (17, 8), (19, 8),
                   (3, 9), (10, 9), (11, 9), (14, 9), (17, 9), (7, 10), (11, 10), (14, 10), (21, 10), (29, 10), (11, 11),
                   (14, 11), (16, 11), (21, 11), (1, 12), (11, 12), (12, 12), (28, 12), (29, 13), (30, 13), (9, 14), (11, 14),
                   (12, 14), (13, 14), (18, 15), (5, 16), (14, 16), (15, 17), (19, 17), (26, 17), (14, 18), (15, 18), (17, 18), (19, 19)]

# Compute Euclidean distance between all pairs of points
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

distance_threshold = 4

# Create a model
model = gp.Model("AED_Placement_Minimal")

# Decision variables
x = model.addVars(all_points, vtype=GRB.BINARY, name="x")

# Objective: Minimize the number of AEDs
model.setObjective(gp.quicksum(x[p] for p in all_points), GRB.MINIMIZE)

# Constraints to ensure each specific point is covered by at least one AED within the distance threshold
for q in specific_points:
    model.addConstr(gp.quicksum(x[p] for p in all_points if euclidean_distance(p, q) <= distance_threshold) >= 1, name=f"cover_{q}")

# Optimize the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print("Minimum number of AEDs needed:", model.objVal)
    print("Locations to place AEDs:")
    aed_locations = [p for p in all_points if x[p].X > 0.5]
    print(aed_locations)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.title('Minimal AED Placement for Full Coverage')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Plot AED locations
    for aed in aed_locations:
        plt.plot(aed[0], aed[1], 'P', color='gold', markersize=15, label='AED Locations' if aed == aed_locations[0] else "")

    # Plot specific points
    for point in specific_points:
        plt.plot(point[0], point[1], 'o', color='blue', label='Specific Points' if point == specific_points[0] else "")

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # removing duplicate labels
    plt.legend(by_label.values(), by_label.keys())

    plt.xlim(-1, 30)
    plt.ylim(-1, 19)
    plt.xticks(range(0, 30))
    plt.yticks(range(0, 19))
    plt.show()

