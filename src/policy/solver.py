from src.policy.base import BasePolicy
import numpy as np
import pulp
from scipy.spatial import distance_matrix


class SolverPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = pulp.PULP_CBC_CMD(threads=24)

    def solve(self, nodes: np.ndarray):
        num_nodes = nodes.shape[0]
        problem = pulp.LpProblem("TSP", pulp.LpMinimize)
        dist_mat = distance_matrix(nodes, nodes)

        x = pulp.LpVariable.dicts(
            "x",
            [(i, j) for i in range(num_nodes) for j in range(num_nodes)],
            cat="Binary",
        )
        u = pulp.LpVariable.dicts(
            "u", (i for i in range(num_nodes)), lowBound=1, upBound=num_nodes, cat="Integer"
        )

        # objective function
        problem += pulp.lpSum(
            dist_mat[i, j] * x[(i, j)] for i in range(num_nodes) for j in range(num_nodes)
        )

        # constraints
        for i in range(num_nodes):
            problem += x[i, i] == 0

        for i in range(num_nodes):
            problem += pulp.lpSum(x[(i, j)] for j in range(num_nodes)) == 1
            problem += pulp.lpSum(x[(j, i)] for j in range(num_nodes)) == 1

        # estimate sub-tour
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and i != 0 and j != 0:
                    problem += u[i] - u[j] <= num_nodes * (1 - x[i, j]) - 1

        problem.solve(solver=self.solver)
        routes = [
            (i, j) for i in range(num_nodes) for j in range(num_nodes) if pulp.value(x[i, j]) == 1
        ]

        return routes


if __name__ == "__main__":
    x = np.random.rand(10, 2)
    policy = SolverPolicy()
    routes = policy.solve(x)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    print(sorted(routes, key=lambda x: x[0]))
