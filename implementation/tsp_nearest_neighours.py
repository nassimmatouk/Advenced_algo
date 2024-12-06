import sys
import time
from helper.useful import generate_random_matrix


class NearestNeighborTSPSolution:
    def __init__(self, adj):
        # Initializing the number of cities, the adjacency matrix (distance matrix),
        # the final path to store the tour, and the total cost of the path
        self.N = len(adj)  # Number of cities
        self.adj = adj  # Adjacency matrix (distance matrix)
        self.final_path = []  # Stores the final path of the tour
        self.cost = 0  # Total cost of the path

    def nearest_neighbor(self, start):
        # List to keep track of the cities that have been visited
        visited = [False] * self.N
        # Start the path with the initial city
        path = [start]
        visited[start] = True  # Mark the starting city as visited
        current_city = start
        total_cost = 0  # Total cost of the tour

        # Loop to visit all cities, except the starting city
        for _ in range(self.N - 1):
            nearest_city = None
            min_distance = sys.maxsize  # Initialize with the maximum possible value

            # Look for the nearest unvisited city
            for city in range(self.N):
                if not visited[city] and self.adj[current_city][city] < min_distance:
                    nearest_city = city  # Update the nearest city
                    min_distance = self.adj[current_city][city]  # Update the minimum distance

            # Add the nearest city to the path
            path.append(nearest_city)
            visited[nearest_city] = True  # Mark the nearest city as visited
            total_cost += min_distance  # Add the distance to the total cost
            current_city = nearest_city  # Move to the next city

        # Finally, return to the starting city to complete the cycle
        total_cost += self.adj[current_city][start]
        path.append(start)

        # Store the final path and the total cost
        self.final_path = path
        self.cost = total_cost

    def solve(self, start=0):
        # Solve the TSP by using the nearest neighbor algorithm
        self.nearest_neighbor(start)
        return self.cost, self.final_path


# Function to measure the execution time of the nearest neighbor algorithm
def measure_execution_time_nearest_neighbor(distances):
    start_time = time.time()  # Start the timer
    solver = NearestNeighborTSPSolution(distances)  # Create an instance of the solver
    cost, path = solver.solve(start=0)  # Solve the problem starting from city 0
    end_time = time.time()  # Stop the timer
    execution_time = end_time - start_time  # Calculate the execution time
    return cost, path, execution_time


# Main code specific to only the nearest neighbor algorithm
if __name__ == "__main__":
    # Loop to test with different numbers of cities (from 3 to 100)
    for number_cities in range(3, 100):
        print("******************** Number of cities:", number_cities, "********************")

        # Generate a random adjacency matrix (distance matrix) for the graph
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)

        # Measure the execution time and get the TSP cost and path
        cost, final_path, execution_time = measure_execution_time_nearest_neighbor(adj)

        # Display the results
        print("Total cost:", cost)
        print("Path taken:", end=' ')
        for city in final_path:
            print(city, end=' ')
        print("\nExecution time:", execution_time)
