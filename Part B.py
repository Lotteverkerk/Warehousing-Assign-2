import numpy as np

# Set grid dimensions
rows = 30
cols = 19

# Create a list of all grid points
points = [(x, y) for x in range(rows) for y in range(cols)]

# Initialize the distance matrix
distance_matrix = np.zeros((rows * cols, rows * cols))

# Compute Euclidean distance between each pair of points
for i in range(len(points)):
    for j in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[j]
        distance_matrix[i, j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Print a small part of the matrix as example
print(distance_matrix[:5, :5])  # Displaying only a small part to check

treshold = 4

a_ij_matrix = (distance_matrix <= treshold).astype(int)

# Example: Check coverage from (7, 10) to (5, 9)
index_i = points.index((7, 10))
index_j = points.index((18, 18))

# Check if a_ij is 1 (covered) or 0 (not covered)
covered = a_ij_matrix[index_i, index_j]
print(f"Point (7, 10) covers Point (5, 9): {'Yes' if covered == 1 else 'No'}")
