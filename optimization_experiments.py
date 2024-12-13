from util_v2 import *
from solver import *
import matplotlib
matplotlib.use('Agg')  # for Linux (not needed for Mac I believe)
import matplotlib.pyplot as plt
import time
import math

def get_covered_tuple_nodes(G, tuple_nodes):
    covered_tuple_nodes = []
    for tuple_node in tuple_nodes:
        if G.degree(tuple_node) > 0:
            covered_tuple_nodes.append(tuple_node)
    
    return covered_tuple_nodes

num_satellites_list = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150]
num_timesteps_list = list(num_satellites_list)
num_locations_list = [5 for _ in range(len(num_satellites_list))]
COVERAGE_PROB = 0.5

x_vals = []
brute_force_times = []
deg_times = []
cost_times = []
ratio_times = []
online_ratio_times = []
ilp_times = []
lp_times = []
deg_gaps = []
cost_gaps = []
ratio_gaps = []
online_ratio_gaps = []
ilp_gaps = []
lp_gaps = []

for (num_timesteps, num_locations, num_satellites) in zip(num_timesteps_list, num_locations_list, num_satellites_list):
    print(f"num_timesteps: {num_timesteps}, num_locations: {num_locations}, num_satellites: {num_satellites}")
    x_vals.append(num_satellites)
    feasible_tuple_nodes = []
    while len(feasible_tuple_nodes) < COVERAGE_PROB * num_timesteps * num_locations:
        G, tuple_nodes, satellite_nodes = create_satellite_bipartite_graph(num_locations, num_timesteps, num_satellites, COVERAGE_PROB)
        feasible_tuple_nodes = get_covered_tuple_nodes(G, tuple_nodes)
    start = time.time()
    ilp_satellite_set, ilp_cost = weighted_set_cover_ilp(G, feasible_tuple_nodes, satellite_nodes)
    end = time.time()
    ilp_times.append(end - start)
    brute_force_cost = ilp_cost
    brute_force_satellite_set = ilp_satellite_set
    '''
    start = time.time()
    brute_force_satellite_set, brute_force_cost = brute_force_algorithm(G, feasible_tuple_nodes, satellite_nodes)
    end = time.time()
    brute_force_times.append(end - start)
    '''
    start = time.time()
    greedy_degree_satellite_set, greedy_degree_cost = greedy_degree_based_algorithm(G, feasible_tuple_nodes, satellite_nodes)
    end = time.time()
    if greedy_degree_cost < brute_force_cost:
        print("SOMETHING WENT WRONG!")
        print(f"Brute force cost: {brute_force_cost}")
        print(f"Greedy degree cost: {greedy_degree_cost}")
        print(f"Satellite nodes: {satellite_nodes}")
        print(f"Greedy degree satellite set: {greedy_degree_satellite_set}")
        print(f"Brute force satellite set: {brute_force_satellite_set}")
    deg_times.append(end - start)
    deg_gaps.append(greedy_degree_cost - brute_force_cost)

    start = time.time()
    greedy_cost_satellite_set, greedy_cost_cost = greedy_cost_based_algorithm(G, feasible_tuple_nodes, satellite_nodes)
    end = time.time()
    if greedy_cost_cost < brute_force_cost:
        print("SOMETHING WENT WRONG!")
        print(f"Brute force cost: {brute_force_cost}")
        print(f"Greedy cost cost: {greedy_cost_cost}")
        print(f"Satellite nodes: {satellite_nodes}")
        print(f"Greedy cost satellite set: {greedy_cost_satellite_set}")
        print(f"Brute force satellite set: {brute_force_satellite_set}")
    cost_times.append(end - start)
    cost_gaps.append(greedy_cost_cost - brute_force_cost)

    start = time.time()
    greedy_ratio_satellite_set, greedy_ratio_cost = greedy_ratio_based_algorithm(G, feasible_tuple_nodes, satellite_nodes)
    end = time.time()
    if greedy_ratio_cost < brute_force_cost:
        print("SOMETHING WENT WRONG!")
        print(f"Brute force cost: {brute_force_cost}")
        print(f"Greedy cost cost: {greedy_cost_cost}")
        print(f"Satellite nodes: {satellite_nodes}")
        print(f"Greedy cost satellite set: {greedy_cost_satellite_set}")
        print(f"Brute force satellite set: {brute_force_satellite_set}")
    ratio_times.append(end - start)
    ratio_gaps.append(greedy_ratio_cost - brute_force_cost)

    '''
    start = time.time()
    online_greedy_ratio_satellite_set, online_greedy_ratio_cost = online_greedy_ratio_based_algorithm(G, feasible_tuple_nodes, satellite_nodes)
    end = time.time()
    if online_greedy_ratio_cost < brute_force_cost:
        print("SOMETHING WENT WRONG!")
        print(f"Brute force cost: {brute_force_cost}")
        print(f"Greedy cost cost: {greedy_cost_cost}")
        print(f"Satellite nodes: {satellite_nodes}")
        print(f"Greedy cost satellite set: {greedy_cost_satellite_set}")
        print(f"Brute force satellite set: {brute_force_satellite_set}")
    online_ratio_times.append(end - start)
    online_ratio_gaps.append(online_greedy_ratio_cost - brute_force_cost)
    '''

    k = 2 * math.ceil(np.log(num_satellites))
    start = time.time()
    lp_satellite_set, lp_cost = weighted_set_cover_lp_relaxation(G, feasible_tuple_nodes, satellite_nodes, k)
    end = time.time()   
    lp_times.append(end - start)
    lp_gaps.append(lp_cost - brute_force_cost)

plt.style.use('classic')
plt.rcParams.update({'font.size': 14})
# plt.plot(x_vals, brute_force_times, marker='s', lw=3, label="Brute-Force")
plt.plot(x_vals, ilp_times, marker='d', lw=3, label="ILP")
plt.plot(x_vals, lp_times, marker='p', lw=3, label="LP-Approx")
plt.plot(x_vals, deg_times, marker='o', lw=3, label="Degree-Greedy")
plt.plot(x_vals, cost_times, marker='x', lw=3, label="Cost-Greedy")
plt.plot(x_vals, ratio_times, marker='*', lw=3, label="Ratio-Greedy")
# plt.plot(x_vals, online_ratio_times, marker='^', lw=3, label="Online Ratio-Greedy")
plt.xlabel("Number of Satellites", fontsize=18)
plt.ylabel("Time (s)", fontsize=18)
plt.legend(loc=(0, 1), frameon=False, ncol=3, fontsize=14)   
plt.savefig('time_comparison.png', bbox_inches='tight')

plt.clf()
plt.rcParams.update({'font.size': 14})
plt.plot(x_vals, lp_gaps, marker='p', lw=3, label="LP-approx")
plt.plot(x_vals, deg_gaps, marker='^', lw=3, label="Degree-Greedy")
plt.plot(x_vals, cost_gaps, marker='v', lw=3, label="Cost-Greedy")
plt.plot(x_vals, ratio_gaps, marker='d', lw=3, label="Ratio-Greedy")
# plt.plot(x_vals, online_ratio_gaps, marker='o', lw=3, label="Online Ratio-Greedy")
plt.xlabel("Number of Satellites", fontsize=18)
plt.ylabel("Optimality Gap (Cost)", fontsize=18)
plt.legend(loc=(0.15, 1), frameon=False, ncols=2, fontsize=14)
plt.savefig('optimality_gap.png', bbox_inches='tight')