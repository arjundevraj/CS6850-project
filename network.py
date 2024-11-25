import matplotlib.pyplot as plt
import argparse
import time
from util import *

parser = argparse.ArgumentParser(
    prog='Satellite Cost Optimization',
    description='Find the optimal combination of satellite plans given coverage requirements'
)

parser.add_argument('--locations', type=int, help='Number of locations', required=True)
parser.add_argument('--timesteps', type=int, help='Number of timesteps', required=True)
parser.add_argument('--satellites', type=int, help='Number of satellites', required=True)
parser.add_argument('--coverage_prob', type=float, help='Coverage probability for each satellite', default=0.9)
parser.add_argument('--min_cost', type=int, help='Minimum cost for a satellite', default=1)
parser.add_argument('--max_cost', type=int, help='Maximum cost for a satellite', default=10)
parser.add_argument('--print-all', type=bool, help='Whether to print all possible combinations', default=False)
parser.add_argument('--visualize', type=bool, help='Whether to visualize the complete coverage graph', default=False)

args = parser.parse_args()
locations = args.locations
timesteps = args.timesteps
satellites = args.satellites
coverage_prob = args.coverage_prob
min_cost = args.min_cost
max_cost = args.max_cost
print_all = args.print_all
visualize = args.visualize

# Create graph
G, tuple_nodes, satellite_nodes = create_satellite_bipartite_graph(
    locations, timesteps, satellites, coverage_prob, min_cost, max_cost
)

# Find all valid coverages
start_time = time.time()
valid_coverages = find_all_valid_coverages(G, tuple_nodes, satellite_nodes)
end_time = time.time()

# Print results
if print_all:
    print("\nAll valid satellite combinations (sorted by total cost):")
    for idx, (satellite_set, total_cost, coverage_details) in enumerate(valid_coverages, 1):
        print(f"\nSolution {idx}:")
        print(f"Satellites used: {satellite_set}")
        print(f"Total cost: {total_cost}")
        print("Coverage details:")
        for loc_time, satellites in coverage_details.items():
            print(f"  {loc_time} covered by:")
            for sat, cost in satellites:
                print(f"    - {sat} at cost {cost}")

print("\nBrute force algorithm results:")
print("Min cost satellite combination:")
satellite_set, total_cost, coverage_details = valid_coverages[0]
print(f"Satellites used: {satellite_set}")
print(f"Total cost: {total_cost}")
print("Coverage details:")
for loc_time, satellites in coverage_details.items():
    print(f"  {loc_time} covered by:")
    for sat in satellites:
        print(f"    - {sat} at cost {satellite_nodes[sat]}")

naive_time = (end_time - start_time) * 1000
print(f"\nSatellite costs: {satellite_nodes}")
print(f"Total number of valid solutions: {len(valid_coverages)}")
print(f"Time taken: {naive_time} ms")

# Run greedy algorithm
start_time = time.time()
greedy_satellite_set, greedy_total_cost, greedy_coverage_details = greedy_degree_based_algorithm(G, tuple_nodes, satellite_nodes)
end_time = time.time()

print("\nGreedy algorithm results:")
print(f"Satellites used: {greedy_satellite_set}")
print(f"Total cost: {greedy_total_cost}")
print("Coverage details:")
for loc_time, satellites in greedy_coverage_details.items():
    print(f"  {loc_time} covered by:")
    for sat in satellites:
        print(f"    - {sat} at cost {satellite_nodes[sat]}")
greedy_time = (end_time - start_time) * 1000
print(f"Time taken: {greedy_time} ms")   
print(f"Time improvement: {naive_time - greedy_time} ms")
print(f"Cost optimality gap: {greedy_total_cost - total_cost}")

if visualize:
    # Visualize the complete coverage
    plt = visualize_coverage(G, tuple_nodes, satellite_nodes)
    plt.savefig('complete_coverage_graph.png')
    plt.show()