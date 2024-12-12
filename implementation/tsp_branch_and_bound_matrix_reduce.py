import sys
import time
from helper.useful import generate_random_matrix, process_memory


class BrandAndBoundTSPSolution:
    def __init__(self, adj):
        # Initialize the solution with the adjacency matrix
        self.N = len(adj)  # Number of cities
        self.adj = adj  # Adjacency matrix
        self.final_path = [None] * (self.N + 1)  # Stores the final solution path
        self.visited = [False] * self.N  # Keeps track of visited cities
        self.cost = sys.maxsize  # The minimum cost (initially set to the maximum value)

    def copy_to_final(self, curr_path):
        # Copy the current path to the final path
        for index in range(self.N):
            self.final_path[index] = curr_path[index]
        self.final_path[self.N] = curr_path[0]  # Completing the tour (return to the starting point)

    def first_min(self, i):
        # Find the smallest edge weight in the row i
        min_val = sys.maxsize
        for k in range(self.N):
            if self.adj[i][k] < min_val and i != k:  # Ignore diagonal (self loop)
                min_val = self.adj[i][k]
        return min_val

    def second_min(self, i):
        # Find the second smallest edge weight in the row i
        first, second = sys.maxsize, sys.maxsize
        for j in range(self.N):
            if i == j:  # Ignore diagonal
                continue
            if self.adj[i][j] <= first:
                second = first
                first = self.adj[i][j]
            elif self.adj[i][j] <= second and self.adj[i][j] != first:
                second = self.adj[i][j]
        return second

    def reduce_matrix(self):
        # Apply matrix reduction (subtract minimum values from rows and columns)
        row_reduced = [min(row) for row in self.adj]  # Reduce rows
        col_reduced = [min(col) for col in zip(*self.adj)]  # Reduce columns

        reduced_adj = []
        for i in range(self.N):
            # Create the reduced matrix
            reduced_adj.append([self.adj[i][j] - row_reduced[i] - col_reduced[j]
                                if self.adj[i][j] != sys.maxsize else sys.maxsize
                                for j in range(self.N)])
        return reduced_adj

    def tsp_rec(self, curr_bound, curr_weight, level, curr_path):
        # Recursive function for Branch and Bound
        if level == self.N:
            # All cities have been visited, check if the tour is complete
            if self.adj[curr_path[level - 1]][curr_path[0]] != 0:
                curr_res = curr_weight + self.adj[curr_path[level - 1]][curr_path[0]]
                if curr_res < self.cost:
                    self.copy_to_final(curr_path)
                    self.cost = curr_res  # Update the minimum cost
            return

        for i in range(self.N):
            if self.adj[curr_path[level - 1]][i] != 0 and not self.visited[i]:
                # Perform branch and bound pruning
                temp = curr_bound
                curr_weight += self.adj[curr_path[level - 1]][i]

                # Apply the matrix reduction to update the bound
                if level == 1:
                    curr_bound -= (self.first_min(curr_path[level - 1]) + self.first_min(i)) / 2
                else:
                    curr_bound -= (self.second_min(curr_path[level - 1]) + self.first_min(i)) / 2

                # If the new bound is less than the current cost, continue the search
                if curr_bound + curr_weight < self.cost:
                    curr_path[level] = i
                    self.visited[i] = True
                    self.tsp_rec(curr_bound, curr_weight, level + 1, curr_path)

                # Backtrack
                curr_weight -= self.adj[curr_path[level - 1]][i]
                curr_bound = temp
                self.visited = [False] * self.N
                for j in range(level):
                    self.visited[curr_path[j]] = True

    def tsp_branch_and_bound(self):
        # Initialize the path and cost, and apply the branch and bound algorithm
        curr_path = [-1] * (self.N + 1)
        curr_bound = 0

        # Apply matrix reduction to get the initial lower bound
        reduced_adj = self.reduce_matrix()

        # Calculate the initial bound based on the reduced matrix
        for index in range(self.N):
            curr_bound += (self.first_min(index) + self.second_min(index))

        # Adjust the bound based on whether it's even or odd
        curr_bound = curr_bound // 2 if curr_bound % 2 == 0 else curr_bound // 2 + 1

        # Start with the first city (0)
        self.visited[0] = True
        curr_path[0] = 0

        # Begin the recursive Branch and Bound search
        self.tsp_rec(curr_bound, 0, 1, curr_path)

        return self.cost, self.final_path


# Function to measure execution time and memory usage
def measure_execution_time_branch_bound(distances):
    start_time = time.time()  # Start the timer
    solver = BrandAndBoundTSPSolution(distances)
    cost, path = solver.tsp_branch_and_bound()  # Run the Branch and Bound solver
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time  # Calculate the execution time
    return cost, path, execution_time


# Main code
if __name__ == "__main__":

    for number_cities in range(3, 100):
        # Generate a random symmetric matrix for the distances between cities
        print("******************** Number of cities:", number_cities, "********************")
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)
        cost, final_path, execution_time = measure_execution_time_branch_bound(adj)  # Measure the execution
        print("Minimum cost:", cost)
        print("Path taken:", end=' ')
        print("Execution time:", execution_time)
        for i in range(len(final_path)):
            print(final_path[i], end=' ')  # Print the final path


