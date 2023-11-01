'''Variables and functions definition'''
#This section imports the necessary libraries and sets up the environment.
import math
import random
import matplotlib.pyplot as plt

'''Function to calculate the distance between 2 points'''
def distance(x,y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

#This block initializes the number of addresses and generates random delivery addresses within the specified range.
num_addresses = 10
max_coordinate_value = 20
addresses = [(random.randint(1, max_coordinate_value), random.randint(1, max_coordinate_value)) for _ in range(num_addresses)]

'''Function to find the nearest unvisited address'''
#This function finds the nearest unvisited address from the current position.
def find_nearest_address(current, unvisited):
    min_distance = float('inf')
    nearest_address = None
    for address in unvisited:
        dist = distance(current, address)
        if dist < min_distance:
            min_distance = dist
            nearest_address = address
    return nearest_address, min_distance

'''Initialize variables for the main function'''
'''This block initializes variables for the main function, 
including the unvisited addresses, the current route, the order of stops, 
the total distance, and the starting position.
It finds the nearest address from the starting point, updates the total distance,
and appends the nearest address to the route'''

unvisited = addresses[:]
route = [(0, 0)] # Start at the depot
order_of_stops = [1]
total_distance = 0
current_position = (0, 0)  

nearest, dist = find_nearest_address(current_position, unvisited)
total_distance += dist
route.append(nearest)
order_of_stops.append(len(route))
current_position = nearest
unvisited.remove(nearest)

'''Main function'''
#This block is the main loop that continues until all addresses are visited. 
#It finds the nearest address, updates the total distance, and appends the nearest address to the route.

while unvisited:
    nearest, dist = find_nearest_address(current_position, unvisited)
    total_distance += dist
    route.append(nearest)
    order_of_stops.append(len(route))
    current_position = nearest
    unvisited.remove(nearest)

'''Return to the depot'''
#This block calculates the distance back to the depot, appends the depot to the route, and updates the order of stops.

total_distance += distance(current_position, (0, 0))
route.append((0, 0))
order_of_stops.append(len(route))

'''Plotting the route with delivery order'''
x_values = [point[0] for point in route]
y_values = [point[1] for point in route]
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values,color='red', marker='o', linestyle='-')
for i, txt in enumerate(order_of_stops):
    plt.annotate(txt, (x_values[i], y_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Optimum Route for Delivery Addresses')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()

'''Printing the results of the code'''
print("Route to fulfill all deliveries using the shortest path:")
for i, point in enumerate(route):
    print(f"{order_of_stops[i]} - {point}")
print("Total Distance:", total_distance)