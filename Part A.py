import gurobipy as gp
import math
from gurobipy import GRB
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

# Define candidate AEDs
candidate_AEDs = [(2,2),(2,6),(3, 4),(3, 5),(3, 6),(3, 9), (4, 6), (5, 5), (5, 6), (5, 8), (6, 8), (7, 2), (7, 3),(7,6)]
candidate_locations = [(9, 7), (5, 2), (5, 3)]

# Create a coverage matrix a_ij
a_ij = np.zeros((len(candidate_AEDs), len(candidate_locations)), dtype=int)

# Check coverage
for i, AED in enumerate(candidate_AEDs):
    for j, location in enumerate(candidate_locations):
        x1, y1 = AED
        x2, y2 = location
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance <= 4:
            a_ij[i, j] = 1

# Print the coverage matrix
print("Coverage Matrix (a_ij):")
print(a_ij)

import matplotlib.pyplot as plt

# Define grid dimensions and spacing
rows = 11
columns = 11
spacing = 1


# Create a grid of points
grid_points = []
for i in range(rows):
    for j in range(columns):
        x = j * spacing
        y = i * spacing
        grid_points.append((x, y))

# Convert candidate AEDs and candidate locations to grid indices
candidate_AED_indices = [(int(y / spacing), int(x / spacing)) for x, y in candidate_AEDs]
candidate_location_indices = [(int(y / spacing), int(x / spacing)) for x, y in candidate_locations]

# Plot the grid
plt.figure(figsize=(8, 6))
for point in grid_points:
    plt.plot(point[0], point[1], 'bo', markersize=6)
# Plot candidate AEDs
for AED in candidate_AED_indices:
    plt.plot(AED[1] * spacing, AED[0] * spacing, 'rs', markersize=10)

# Plot candidate locations
for location in candidate_location_indices:
    plt.plot(location[1] * spacing, location[0] * spacing, 'g^', markersize=10)

# Add labels and title
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Candidate AEDs and Locations on Grid')

# Set aspect ratio and grid
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# Show plot
plt.legend(['Grid Points', 'Candidate AEDs', 'Candidate Locations'], loc='upper right')
plt.show()
'''

# Record the start time to calculate the computation time of the model
start_time = time.time()

# Create a new model
model = gp.Model("M1")

# Sets and Indices
I =                             # Set of candidate locations for new AEDs
J =                             # Set of simulated cardiac arrests to be covered.

# Parameters
Q = truck_cap                   # Capacity of the vehicle
C_t = order_revenue             # Transportation costs in euros/hour of driving
C_e = 0.05                       # Emission costs in euros/kg of CO2



# Decision Variables
x = model.addVars(V, V, K, vtype=GRB.BINARY, name="x")  # Binary variable indicating if vehicle travels from i to j


# Objective Function
model.setObjective(gp.quicksum(R * d[i] for i in P) - gp.quicksum(x[i, j, k] * (C_t * t_ij[i][j] + C_e * E * d_ij[i][j]) 
                   for i in V for j in V for k in K), GRB.MAXIMIZE)
# Constraints

# 2. All pick-up nodes should be visited
for i in P:
    model.addConstr(gp.quicksum(x[i, j, k] for j in N if j != i for k in K)  == 1, "2.Constraint[%s]" % i)

# 3. The delivery location is visited if the pickup location is visited and that the visit is performed by the same vehicle.
for k in K:
    for i in P:
        model.addConstr(gp.quicksum(x[i, j, k] for j in V) - gp.quicksum(x[j, n + i, k] for j in V) == 0, "3.Constraint[%d,%d]" % (k, i))
 


# Set time limit (in seconds)
time_limit = 600  # 10 minutes

# Set time limit parameter
model.setParam('TimeLimit', time_limit)

# Optimize the model
model.optimize()

# Check solution status
if model.status == gp.GRB.Status.INFEASIBLE:
    print("The model is infeasible.")
    # You can print more detailed information about infeasible constraints if needed
    model.computeIIS()
    model.write("infeasible_constraints.ilp")
elif model.status == gp.GRB.Status.OPTIMAL:
    print("The model is optimal.")
else:
    print("The model is infeasible or unbounded.")

if model.Status == GRB.TIME_LIMIT:
    # Provide the best known solution and its corresponding gap
    best_solution = model.objVal  # Best known solution
    gap = model.MIPGap  # Gap between the best known solution and the proven optimal solution
    print(f"Best known solution: {best_solution}, Gap: {gap}")

# Record the end time to calculate the computation time of the model
end_time = time.time()

# Calculate the elapsed time of the the computation time of the model
elapsed_time = end_time - start_time
'''


'''
import matplotlib.pyplot as plt

# Define grid dimensions and spacing
rows = 9
columns = 10
spacing = 1

# Define candidate AEDs and candidate locations
candidate_AEDs = [(3, 1), (5, 2), (5, 3)]
candidate_locations = [(2, 2), (5, 2), (5, 3)]

# Create a grid of points
grid_points = []
for i in range(rows):
    for j in range(columns):
        x = j * spacing
        y = i * spacing
        grid_points.append((x, y))

# Convert candidate AEDs and candidate locations to grid indices
candidate_AED_indices = [(int(y / spacing), int(x / spacing)) for x, y in candidate_AEDs]
candidate_location_indices = [(int(y / spacing), int(x / spacing)) for x, y in candidate_locations]

# Plot the grid
plt.figure(figsize=(8, 6))
for point in grid_points:
    plt.plot(point[0], point[1], 'bo', markersize=6)
# Plot candidate AEDs
for AED in candidate_AED_indices:
    plt.plot(AED[1] * spacing, AED[0] * spacing, 'rs', markersize=10)

# Plot candidate locations
for location in candidate_location_indices:
    plt.plot(location[1] * spacing, location[0] * spacing, 'g^', markersize=10)

# Add labels and title
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Candidate AEDs and Locations on Grid')

# Set aspect ratio and grid
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# Show plot
plt.legend(['Grid Points', 'Candidate AEDs', 'Candidate Locations'], loc='upper right')
plt.show()
'''


'''
# Define grid dimensions and spacing
rows = 9
columns = 10
spacing = 25

# Generate grid points
grid_points = []
for i in range(rows):
    for j in range(columns):
        x = j * spacing
        y = i * spacing
        grid_points.append((x, y))


# Calculate Euclidean distance matrix
num_points = len(grid_points)
distance_matrix = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        x1, y1 = grid_points[i]
        x2, y2 = grid_points[j]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_matrix[i, j] = distance
        
# Print distance matrix
print("Euclidean Distance Matrix:")
print(distance_matrix)

# Create binary matrix indicating distances less than or equal to 100
dij_a = np.where(distance_matrix <= 100, 1, 0)

# Print binary matrix
print("Binary Matrix (dij_a) indicating distances <= 100:")
print(dij_a)

'''