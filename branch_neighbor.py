'''Variables and functions definition'''
# This section imports the necessary libraries and sets up the environment.
import math
import random
import matplotlib.pyplot as plt
'''Function to calculate the distance between 2 points'''
def distance(x,y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

# This block initializes the number of addresses and generates random delivery addresses within the specified range.
num_addresses = 10
max_coordinate_value = 20
addresses = [(random.randint(1, max_coordinate_value), random.randint(1, max_coordinate_value)) for _ in range(num_addresses)]

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

'''Plotting the results'''
x_values_nn = [point[0] for point in route_nn]
y_values_nn = [point[1] for point in route_nn]

x_values_bb = [point[0] for point in shortest_route]
y_values_bb = [point[1] for point in shortest_route]


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_values_nn, y_values_nn, color='blue', marker='o', linestyle='-', label='Nearest Neighbor')
for i, txt in enumerate(order_of_stops_nn):
    plt.annotate(txt, (x_values_nn[i], y_values_nn[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Nearest Neighbor Algorithm')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

plt.subplot(1, 2, 2)
plt.plot(x_values_bb, y_values_bb, color='green', marker='o', linestyle='-', label='Branch and Bound')
for i, txt in enumerate(range(len(shortest_route))):
    plt.annotate(i, (x_values_bb[i], y_values_bb[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Branch and Bound Algorithm')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')


plt.show()