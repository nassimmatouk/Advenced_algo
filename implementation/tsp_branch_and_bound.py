import sys
import time

import numpy as np

from helper.useful import generate_random_matrix


class BrandAndBoundTSPSolution:
    def __init__(self, adj):
        self.N = len(adj)
        self.adj = adj
        self.final_path = [None] * (self.N + 1)
        self.visited = [False] * self.N
        self.final_res = sys.maxsize

    def copy_to_final(self, curr_path):
        for index in range(self.N):
            self.final_path[index] = curr_path[index]
        self.final_path[self.N] = curr_path[0]

    def first_min(self, i):
        min_val = sys.maxsize
        for k in range(self.N):
            if self.adj[i][k] < min_val and i != k:
                min_val = self.adj[i][k]
        return min_val

    def second_min(self, i):
        first, second = sys.maxsize, sys.maxsize
        for j in range(self.N):
            if i == j:
                continue
            if self.adj[i][j] <= first:
                second = first
                first = self.adj[i][j]
            elif self.adj[i][j] <= second and self.adj[i][j] != first:
                second = self.adj[i][j]
        return second

    def tsp_rec(self, curr_bound, curr_weight, level, curr_path):
        if level == self.N:
            if self.adj[curr_path[level - 1]][curr_path[0]] != 0:
                curr_res = curr_weight + self.adj[curr_path[level - 1]][curr_path[0]]
                if curr_res < self.final_res:
                    self.copy_to_final(curr_path)
                    self.final_res = curr_res
            return

        for i in range(self.N):
            if self.adj[curr_path[level - 1]][i] != 0 and not self.visited[i]:
                temp = curr_bound
                curr_weight += self.adj[curr_path[level - 1]][i]

                if level == 1:
                    curr_bound -= (self.first_min(curr_path[level - 1]) + self.first_min(i)) / 2
                else:
                    curr_bound -= (self.second_min(curr_path[level - 1]) + self.first_min(i)) / 2

                if curr_bound + curr_weight < self.final_res:
                    curr_path[level] = i
                    self.visited[i] = True
                    self.tsp_rec(curr_bound, curr_weight, level + 1, curr_path)

                curr_weight -= self.adj[curr_path[level - 1]][i]
                curr_bound = temp
                self.visited = [False] * self.N
                for j in range(level):
                    self.visited[curr_path[j]] = True

    def solve(self):
        curr_path = [-1] * (self.N + 1)
        curr_bound = 0

        for index in range(self.N):
            curr_bound += (self.first_min(index) + self.second_min(index))

        curr_bound = curr_bound // 2 if curr_bound % 2 == 0 else curr_bound // 2 + 1

        self.visited[0] = True
        curr_path[0] = 0

        self.tsp_rec(curr_bound, 0, 1, curr_path)

        return self.final_res, self.final_path

def measure_execution_time_branch_bound(distances):
    start_time = time.time()
    solver = BrandAndBoundTSPSolution(distances)
    final_res, final_path = solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return final_res, final_path, execution_time


# Code principal
if __name__ == "__main__":


    for number_cities in range(3, 100):
        print("******************** Nombre de villes :", number_cities, "********************")
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)
        cost, final_path,  execution_time= measure_execution_time_branch_bound(adj)
        print("Coût minimum :", cost)
        print("Chemin emprunté :", end=' ')
        print("Temps d'exécution :", execution_time)
        for i in range(len(final_path)):
            print(final_path[i], end=' ')
