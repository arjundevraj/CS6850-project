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
COVERAGE_PROB = 0.4

x_vals = []
ilp_costs = []
ilp_coverages = []

lambdas = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 5, 10]

for l in lambdas:
    x_vals.append(l)
    feasible_tuple_nodes = []
    while len(feasible_tuple_nodes) < COVERAGE_PROB * NUM_TIMESTEPS * NUM_LOCATIONS:
        G, tuple_nodes, satellite_nodes = create_satellite_bipartite_graph(NUM_LOCATIONS, NUM_TIMESTEPS, NUM_SATELLITES, COVERAGE_PROB)
        feasible_tuple_nodes = get_covered_tuple_nodes(G, tuple_nodes)
    
    all_coverable_tuple_nodes = set([tuple_node for tuple_node in tuple_nodes if G.degree(tuple_node) > 0])
    ilp_satellite_set, ilp_cost = weighted_set_cover_ilp_tradeoff(G, feasible_tuple_nodes, satellite_nodes, l)
    ilp_costs.append(ilp_cost)
    ilp_covered = set()
    for satellite in ilp_satellite_set:
        ilp_covered.update(set(G.neighbors(satellite)))
    ilp_coverages.append(100 * len(ilp_covered) / len(all_coverable_tuple_nodes))

print(ilp_coverages)
'''
plt.style.use('classic')
plt.rcParams.update({'font.size': 14})
plt.plot(x_vals, ilp_costs, marker='.', lw=3)
plt.xlabel("Coverage Probability", fontsize=18)
plt.ylabel("Cost of Optimal Satellite Set", fontsize=18)
plt.savefig('cost_ilp2.png', bbox_inches='tight')

plt.clf()
plt.rcParams.update({'font.size': 14})
plt.plot(x_vals, ilp_coverages, marker='.', lw=3)
plt.ticklabel_format(useOffset=False)
plt.xlabel("Coverage Probability", fontsize=18)
plt.ylabel("Coverage of Optimal Satellite Set (%)", fontsize=18)
plt.savefig('coverage_ilp2.png', bbox_inches='tight')
'''

fig, ax1 = plt.subplots()
plt.style.use('classic')
plt.rcParams.update({'font.size': 14})
# First y-axis (left)
ax1.plot(x_vals, ilp_costs, marker='o', lw=3, label="Cost", color='blue')
ax1.set_xlabel("Lambda", fontsize=17)
ax1.set_ylabel("Cost of Optimal Satellite Set", fontsize=15)
ax1.set_ylim(0, 50)
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_xlim(-1, 11)

# Second y-axis (right)
ax2 = ax1.twinx()
ax2.plot(x_vals, ilp_coverages, marker='s', lw=3, label="Coverage", color='red')
ax2.set_ylabel("Coverage of Optimal Satellite Set (%)", fontsize=15)
ax2.tick_params(axis='y', labelsize=14)
ax2.set_ylim(0, 101)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=14)

# Save the figure
plt.savefig('combined_ilp_lambda.png', bbox_inches='tight')