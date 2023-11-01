import numpy as np
import random
import matplotlib.pyplot as plt

# Function to calculate the distance between two points
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Ant Colony Optimization Algorithm
class AntColony:
    def __init__(self, distances, num_ants, num_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        shortest_distance = np.inf
        for i in range(self.num_iterations):
            all_paths = self.generate_paths()
            self.spread_pheromone(all_paths)
            current_shortest_path, current_shortest_distance = min(all_paths, key=lambda x: x[1])
            if current_shortest_distance < shortest_distance:
                shortest_path = current_shortest_path
                shortest_distance = current_shortest_distance
            self.pheromone *= self.decay
        return shortest_path, shortest_distance

    def spread_pheromone(self, all_paths):
        all_paths_sorted = sorted(all_paths, key=lambda x: x[1])
        for path, dist in all_paths_sorted[:self.num_ants]:
            for move in path:
                if dist != 0:
                    self.pheromone[move] += 1.0 / dist
                else:
                    self.pheromone[move] += 0

    def generate_paths(self):
        all_paths = []
        for _ in range(self.num_ants):
            path = self.generate_path()
            total_distance = sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
            all_paths.append((path, total_distance))
        return all_paths

    def generate_path(self):
        path = [0]  # Start at the depot
        unvisited = set(range(1, len(self.distances)))
        current = 0
        while unvisited:
            move = self.pick_move(current, unvisited)
            path.append(move)
            unvisited.remove(move)
            current = move
        path.append(0)  # Return to the depot
        return path

    def pick_move(self, current, unvisited):
        pheromone_values = np.power(self.pheromone[current, list(unvisited)], self.alpha)
        distance_values = np.power(1.0 / (self.distances[current, list(unvisited)] + 1e-10), self.beta)  # Add small value to avoid division by zero
        probabilities = pheromone_values * distance_values
        probabilities /= probabilities.sum()
        return np.random.choice(list(unvisited), p=probabilities)

# Number of delivery addresses and maximum coordinate value for random generation
num_addresses = 10
max_coordinate_value = 10

# Generate random delivery addresses
addresses = [(random.randint(1, max_coordinate_value), random.randint(1, max_coordinate_value)) for _ in range(num_addresses)]

# Add the depot at the beginning
addresses.insert(0, (0, 0))
num_addresses += 1

# Calculate distance matrix
dist_matrix = np.zeros((num_addresses, num_addresses))
for i in range(num_addresses):
    for j in range(num_addresses):
        dist_matrix[i, j] = distance(addresses[i], addresses[j])

# Create an instance of the Ant Colony Optimization algorithm
colony = AntColony(dist_matrix, num_ants=10, num_iterations=100, decay=0.6, alpha=1, beta=2)

# Run the algorithm and get the shortest path and distance
shortest_path, shortest_distance = colony.run()

# Print the results
print("Shortest Route to fulfill all deliveries:")
for i, point in enumerate(shortest_path):
    print(f"{i} - {addresses[point]}")
print("Total Distance:", shortest_distance)

# Plot the delivery route
x_values = [addresses[p][0] for p in shortest_path]
y_values = [addresses[p][1] for p in shortest_path]

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, marker='o')
plt.plot(x_values[0], y_values[0], marker='s', color='r')  # Mark the depot with a square
for i, txt in enumerate(shortest_path):
    plt.annotate(i, (addresses[txt][0], addresses[txt][1]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.title('Delivery Route')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()
