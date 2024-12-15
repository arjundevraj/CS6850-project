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

NUM_SATELLITES = 50
NUM_TIMESTEPS = 50
NUM_LOCATIONS = 5
coverage_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
deg_coverages = []
cost_coverages = []
ratio_coverages = []
lp_coverages = []
ilp_coverages = []

for coverage_prob in coverage_probs:
    x_vals.append(coverage_prob)
    feasible_tuple_nodes = []
    while len(feasible_tuple_nodes) < coverage_prob * NUM_TIMESTEPS * NUM_LOCATIONS:
        G, tuple_nodes, satellite_nodes = create_satellite_bipartite_graph(NUM_LOCATIONS, NUM_TIMESTEPS, NUM_SATELLITES, coverage_prob)
        feasible_tuple_nodes = get_covered_tuple_nodes(G, tuple_nodes)
    
    all_coverable_tuple_nodes = set([tuple_node for tuple_node in tuple_nodes if G.degree(tuple_node) > 0])
    start = time.time()
    ilp_satellite_set, ilp_cost = weighted_set_cover_ilp(G, feasible_tuple_nodes, satellite_nodes)
    end = time.time()
    ilp_times.append(end - start)
    brute_force_cost = ilp_cost
    brute_force_satellite_set = ilp_satellite_set
    ilp_covered = set()
    for satellite in ilp_satellite_set:
        ilp_covered.update(set(G.neighbors(satellite)))
    ilp_coverages.append(100 * len(ilp_covered) / len(all_coverable_tuple_nodes))
   
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
    deg_covered = set()
    for satellite in greedy_degree_satellite_set:
        deg_covered.update(set(G.neighbors(satellite)))
    deg_coverages.append(100 * len(deg_covered) / len(all_coverable_tuple_nodes))

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
    cost_covered = set()
    for satellite in greedy_cost_satellite_set:
        cost_covered.update(set(G.neighbors(satellite)))
    cost_coverages.append(100 * len(cost_covered) / len(all_coverable_tuple_nodes))

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
    ratio_covered = set()
    for satellite in greedy_ratio_satellite_set:
        ratio_covered.update(set(G.neighbors(satellite)))
    ratio_coverages.append(100 * len(ratio_covered) / len(all_coverable_tuple_nodes))

    k = 2 * math.ceil(np.log(NUM_TIMESTEPS * NUM_LOCATIONS))
    start = time.time()
    lp_satellite_set, lp_cost = weighted_set_cover_lp_relaxation(G, feasible_tuple_nodes, satellite_nodes, k)
    end = time.time()   
    lp_times.append(end - start)
    lp_gaps.append(lp_cost - brute_force_cost)
    lp_covered = set()
    for satellite in lp_satellite_set:
        lp_covered.update(set(G.neighbors(satellite)))
    lp_coverages.append(100 * len(lp_covered) / len(all_coverable_tuple_nodes))

plt.style.use('classic')
plt.rcParams.update({'font.size': 14})
plt.plot(x_vals, deg_gaps, marker='o', lw=3, label="Degree-Greedy")
plt.plot(x_vals, cost_gaps, marker='x', lw=3, label="Cost-Greedy")
plt.plot(x_vals, ratio_gaps, marker='s', lw=3, label="Ratio-Greedy")
plt.plot(x_vals, lp_gaps, marker='p', lw=3, label="LP-Approx")
plt.xlabel("Coverage probability", fontsize=18)
plt.ylabel("Optimality Gap (Cost)", fontsize=18)
plt.legend(loc=(0.15, 1), frameon=False, ncols=2, fontsize=14)
plt.savefig('optimality_gap_prob.png', bbox_inches='tight')