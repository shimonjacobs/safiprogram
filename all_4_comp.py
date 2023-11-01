import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Variables and functions definition
def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

num_addresses = 10
max_coordinate_value = 20
addresses = [(random.randint(1, max_coordinate_value), random.randint(1, max_coordinate_value)) for _ in range(num_addresses)]
# Genetic Algorithm
class GeneticAlgorithmTSP:
    def __init__(self, num_addresses, population_size=100, elite_size=20, mutation_rate=0.01):
        self.num_addresses = num_addresses
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

    def create_route(self, addresses):
        route = random.sample(addresses, len(addresses))
        route.insert(0, (0, 0))  # add address 0 at the beginning
        route.append((0, 0))  # add address 0 at the end
        return route

    def initial_population(self, addresses):
        population = []
        for _ in range(self.population_size):
            population.append(self.create_route(addresses))
        return population

    def rank_routes(self, population):
        return sorted(population, key=lambda x: self.get_route_distance(x))

    def get_route_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance(route[i], route[i + 1])
        return total_distance

    def breed(self, parent1, parent2):
        child = []
        gene1, gene2 = random.sample(range(1, len(parent1) - 1), 2)  # avoid the first and last element
        start_gene = min(gene1, gene2)
        end_gene = max(gene1, gene2)
        for i in range(start_gene, end_gene):
            child.append(parent1[i])
        for item in parent2:
            if item not in child:
                child.append(item)
        return [parent1[0]] + child + [parent1[-1]]

    def mutate(self, route):
        for swapped in range(1, len(route) - 1):  # avoid the first and last element
            if random.random() < self.mutation_rate:
                swap_with = int(random.random() * (len(route) - 2)) + 1  # avoid the first and last element
                address1 = route[swapped]
                address2 = route[swap_with]
                route[swapped] = address2
                route[swap_with] = address1
        return route

    def next_generation(self, current_gen):
        elite = self.rank_routes(current_gen)[:self.elite_size]
        rest = current_gen[self.elite_size:]
        children = []
        for i in range(len(rest)):
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = self.breed(parent1, parent2)
            child = self.mutate(child)
            children.append(child)
        elite.extend(children)
        return elite

# Simulated Annealing Algorithm
def total_distance(route):
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1))

def simulated_annealing(addresses, num_iter=10000, initial_temperature=1000, cooling_rate=0.003):
    current_route = [(0, 0)] + addresses + [(0, 0)]
    best_route = list(current_route)
    min_dist = total_distance(best_route)
    temperature = initial_temperature

    for _ in range(num_iter):
        new_route = list(current_route)
        index1 = random.randint(1, len(addresses))
        index2 = random.randint(1, len(addresses))
        new_route[index1], new_route[index2] = new_route[index2], new_route[index1]

        current_dist = total_distance(current_route)
        new_dist = total_distance(new_route)

        if new_dist < min_dist:
            min_dist = new_dist
            best_route = new_route

        if new_dist < current_dist or random.random() < math.exp((current_dist - new_dist) / temperature):
            current_route = new_route

        temperature *= 1 - cooling_rate

    return best_route, min_dist

# Computing using Genetic Algorithm
genetic_algorithm = GeneticAlgorithmTSP(num_addresses, population_size=100, elite_size=20, mutation_rate=0.01)
initial_population = genetic_algorithm.initial_population(addresses)

for i in range(1000):
    initial_population = genetic_algorithm.next_generation(initial_population)

# Computing using Simulated Annealing
sa_shortest_route, sa_min_dist = simulated_annealing(addresses)
'''Branch and Bound Algorithm'''
def total_distance(route):
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1))

def all_perms(elements):
    if len(elements) <= 1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                yield perm[:i] + elements[0:1] + perm[i:]

def branch_and_bound(addresses):
    shortest_route = None
    min_dist = float('inf')
    for perm in all_perms(addresses):
        route = [(0, 0)] + list(perm) + [(0, 0)]
        dist = total_distance(route)
        if dist < min_dist:
            min_dist = dist
            shortest_route = route
    return shortest_route, min_dist

'''Nearest Neighbor Algorithm'''
unvisited = addresses[:]
route_nn = [(0, 0)]  # Start at the depot
order_of_stops_nn = [1]
total_distance_nn = 0
current_position_nn = (0, 0)

def find_nearest_address(current, unvisited):
    min_distance = float('inf')
    nearest_address = None
    for address in unvisited:
        dist = distance(current, address)
        if dist < min_distance:
            min_distance = dist
            nearest_address = address
    return nearest_address, min_distance

nearest, dist = find_nearest_address(current_position_nn, unvisited)
total_distance_nn += dist
route_nn.append(nearest)
order_of_stops_nn.append(len(route_nn))
current_position_nn = nearest
unvisited.remove(nearest)

while unvisited:
    nearest, dist = find_nearest_address(current_position_nn, unvisited)
    total_distance_nn += dist
    route_nn.append(nearest)
    order_of_stops_nn.append(len(route_nn))
    current_position_nn = nearest
    unvisited.remove(nearest)

total_distance_nn += distance(current_position_nn, (0, 0))
route_nn.append((0, 0))
order_of_stops_nn.append(len(route_nn))

'''Comparing the results'''
shortest_route, min_dist = branch_and_bound(addresses)

print("Nearest Neighbor Algorithm:")
print("Route to fulfill all deliveries using the shortest path:")
for i, point in enumerate(route_nn):
    print(f"{order_of_stops_nn[i]} - {point}")
print("Total Distance:", total_distance_nn)

print("\nBranch and Bound Algorithm:")
print("Shortest Route to fulfill all deliveries:")
for i, point in enumerate(shortest_route):
    print(f"{i} - {point}")
print("Total Distance:", min_dist)

print("\nGenetic Algorithm for TSP:")
print("Shortest Route to fulfill all deliveries:")
for i, point in enumerate(initial_population[0]):
    print(f"{i} - {point}")
print("Total Distance:", genetic_algorithm.get_route_distance(initial_population[0]))

print("\nSimulated Annealing Algorithm for TSP:")
print("Shortest Route to fulfill all deliveries:")
for i, point in enumerate(sa_shortest_route):
    print(f"{i} - {point}")
print("Total Distance:", sa_min_dist)

'''Plotting the results'''
x_values_nn = [point[0] for point in route_nn]
y_values_nn = [point[1] for point in route_nn]

x_values_bb = [point[0] for point in shortest_route]
y_values_bb = [point[1] for point in shortest_route]

x_values_ga = [point[0] for point in initial_population[0]]
y_values_ga = [point[1] for point in initial_population[0]]

x_values_sa = [point[0] for point in sa_shortest_route]
y_values_sa = [point[1] for point in sa_shortest_route]

plt.figure(figsize=(16,6))

plt.subplot(1,4,1)
plt.plot(x_values_nn, y_values_nn, color='blue', marker='o', linestyle='-', label='Nearest Neighbor')
for i, txt in enumerate(order_of_stops_nn):
    plt.annotate(txt, (x_values_nn[i], y_values_nn[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Nearest Neighbor Algorithm')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

plt.subplot(1, 4, 2)
plt.plot(x_values_bb, y_values_bb, color='green', marker='o', linestyle='-', label='Branch and Bound')
for i, txt in enumerate(range(len(shortest_route))):
    plt.annotate(i, (x_values_bb[i], y_values_bb[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Branch and Bound Algorithm')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

plt.subplot(1, 4, 3)
plt.plot(x_values_ga, y_values_ga, color='purple', marker='o', linestyle='-', label='Genetic Algorithm')
for i, txt in enumerate(range(len(initial_population[0]))):
    plt.annotate(i, (x_values_ga[i], y_values_ga[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.title('Genetic Algorithm for TSP')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

plt.subplot(1, 4, 4)
plt.plot(x_values_sa, y_values_sa, color='orange', marker='o', linestyle='-', label='Simulated Annealing')
for i, txt in enumerate(range(len(sa_shortest_route))):
    plt.annotate(i, (x_values_sa[i], y_values_sa[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.title('Simulated Annealing Algorithm for TSP')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

plt.show()