from util import *
import matplotlib.pyplot as plt
import time

def is_valid_graph(G, tuple_nodes):
    for tuple_node in tuple_nodes:
        if G.degree(tuple_node) == 0:
            return False
    
    return True

num_satellites_list = [1, 2, 5, 10, 15, 20]
num_timesteps_list = [1, 2, 5, 10, 15, 20]
num_locations_list = [5 for _ in range(len(num_satellites_list))]

x_vals = []
brute_force_times = []
deg_times = []
cost_times = []
deg_optimality_gaps = []
cost_optimality_gaps = []

for (num_timesteps, num_locations, num_satellites) in zip(num_timesteps_list, num_locations_list, num_satellites_list):
    print(f"num_timesteps: {num_timesteps}, num_locations: {num_locations}, num_satellites: {num_satellites}")
    x_vals.append(num_satellites)
    invalid = True
    while invalid:
        G, tuple_nodes, satellite_nodes = create_satellite_bipartite_graph(num_locations, num_timesteps, num_satellites, 0.9)
        invalid = not is_valid_graph(G, tuple_nodes)
    start = time.time()
    brute_force_satellite_set, brute_force_cost = brute_force_algorithm(G, tuple_nodes, satellite_nodes)
    end = time.time()
    brute_force_times.append(end - start)
    start = time.time()
    greedy_degree_satellite_set, greedy_degree_cost = greedy_degree_based_algorithm(G, tuple_nodes, satellite_nodes)
    end = time.time()
    if greedy_degree_cost < brute_force_cost:
        print("SOMETHING WENT WRONG!")
        print(f"Brute force cost: {brute_force_cost}")
        print(f"Greedy cost: {greedy_degree_cost}")
        print(f"Satellite nodes: {satellite_nodes}")
        print(f"Greedy satellite set: {greedy_degree_satellite_set}")
        print(f"Brute force satellite set: {brute_force_satellite_set}")
    deg_times.append((end - start) * 1000)
    deg_optimality_gaps.append(greedy_degree_cost - brute_force_cost)

    start = time.time()
    greedy_cost_satellite_set, greedy_cost_cost = greedy_cost_based_algorithm(G, tuple_nodes, satellite_nodes)
    end = time.time()
    if greedy_cost_cost < brute_force_cost:
        print("SOMETHING WENT WRONG!")
        print(f"Brute force cost: {brute_force_cost}")
        print(f"Greedy cost: {greedy_cost_cost}")
        print(f"Satellite nodes: {satellite_nodes}")
        print(f"Greedy satellite set: {greedy_cost_satellite_set}")
        print(f"Brute force satellite set: {brute_force_satellite_set}")
    cost_times.append((end - start) * 1000)
    cost_optimality_gaps.append(greedy_cost_cost - brute_force_cost)

plt.style.use('classic')
plt.plot(x_vals, deg_times, marker='o', lw=2.5, label="Degree Greedy")
plt.plot(x_vals, cost_times, marker='x', lw=2.5, label="Cost Greedy")
plt.plot(x_vals, brute_force_times, marker='s', lw=2.5, label="Brute Force")
plt.xlabel("Number of Satellites")
plt.ylabel("Time (ms)")
plt.legend()
plt.savefig('time_comparison.png')

plt.clf()
plt.plot(x_vals, deg_optimality_gaps, marker='^', lw=2.5, label="Degree Greedy")
plt.plot(x_vals, cost_optimality_gaps, marker='v', lw=2.5, label="Cost Greedy")
plt.xlabel("Number of Satellites")
plt.ylabel("Optimality Gap (Cost)")
plt.legend()
plt.savefig('optimality_gap.png')