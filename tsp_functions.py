import numpy as np
import random

def tsp_cost(path, dist):
    n = len(path)
    return sum(dist[path[i], path[(i + 1) % n]] for i in range(n))

def tsp_neighbor(path, dist, alpha):
    n = len(path)
    current_cost = tsp_cost(path, dist)
    probes = n * alpha
    for _ in range(probes):
        i, j = sorted(random.sample(range(n), 2))
        candidate = np.concatenate((path[:i], path[i:j+1][::-1], path[j+1:]))
        cost = tsp_cost(candidate, dist)
        if cost < current_cost:
            return candidate, cost, True
    return path, current_cost, False

def tsp_kick(best_path, dist):
    n = len(best_path)
    # 4-point double-bridge style kick (strong perturbation)
    i, j, k, l = sorted(random.sample(range(n), 4))
    new_path = np.concatenate([
        best_path[:i],
        best_path[j:k+1],
        best_path[l:],
        best_path[k+1:l],
        best_path[i:j]
    ])
    return new_path[:n]

def greedy_tsp(dist):
    n = len(dist)
    path = [0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    for _ in range(1, n):
        last = path[-1]
        unvisited_indices = np.where(~visited)[0]
        if not unvisited_indices.size: break

        distances_to_unvisited = dist[last, unvisited_indices]
        local_next_index = np.argmin(distances_to_unvisited)

        next_city = unvisited_indices[local_next_index]

        path.append(next_city)
        visited[next_city] = True

    final_path = np.array(path)
    return final_path
