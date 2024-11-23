import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations
from collections import defaultdict
import numpy as np
import argparse
import time

def create_satellite_bipartite_graph(locations, timesteps, satellites, coverage_prob, min_cost=1, max_cost=10):
    """Previous function with increased coverage probability"""
    assert coverage_prob >= 0 and coverage_prob <= 1, "Coverage probability must be between 0 and 1"
    G = nx.Graph()
    edge_costs = {}
    
    # Create (location, timestep) tuple nodes
    tuple_nodes = []
    for l in range(locations):
        for t in range(timesteps):
            node = (f'L{l}', f'T{t}')
            tuple_nodes.append(node)
            G.add_node(node, bipartite=0)
            
    # Create satellite nodes
    satellite_nodes = {}
    for s in range(satellites):
        cost = random.randint(min_cost, max_cost)
        satellite_nodes[f'S{s}'] = cost
        G.add_node(f'S{s}', bipartite=1, cost=cost)
    
    for s in satellite_nodes:
        # Each satellite covers more location-time pairs
        num_coverages = int(coverage_prob * len(tuple_nodes))
        covered_tuples = random.sample(tuple_nodes, num_coverages)
        for tuple_node in covered_tuples:
            G.add_edge(tuple_node, s)
    
    return G, tuple_nodes, satellite_nodes

def get_coverage_map(G, tuple_nodes, satellites):
    """
    Creates a mapping of each location-time tuple to all satellites that can cover it.
    
    Returns:
        dict: Mapping of tuple -> list of (satellite, cost) pairs
    """
    coverage_map = defaultdict(list)
    for tuple_node in tuple_nodes:
        for satellite in G.neighbors(tuple_node):
            cost = satellites.get(satellite)
            coverage_map[tuple_node].append((satellite, cost))
    return coverage_map

def find_all_valid_coverages(G, tuple_nodes, satellites):
    """
    Find all valid combinations of satellites that provide full coverage.
    Each location-time tuple can be covered by multiple satellites.
    
    Returns:
        list: List of (satellite_set, total_cost, coverage_details) tuples
    """
    valid_solutions = []
    
    # Check if each location-time tuple has at least one satellite covering it
    for tuple_node in tuple_nodes:
        if G.degree(tuple_node) == 0:
            print(f"Warning: {tuple_node} has no satellite coverage!")
            return []

    # Try all possible combinations of satellites
    for r in range(1, len(satellites) + 1):
        for satellite_subset in combinations(satellites, r):
            satellite_set = set(satellite_subset)
            
            # Check if this combination provides full coverage
            coverage_details = defaultdict(set)
            is_valid = True
            
            # Check each location-time tuple
            for tuple_node in tuple_nodes:
                # Get all satellites from our subset that can cover this tuple
                covering_satellites = set()
                for sat in G.neighbors(tuple_node):
                    if sat in satellite_set:
                        covering_satellites.add(sat)
                
                if not covering_satellites:
                    is_valid = False
                    break
                    
                coverage_details[tuple_node] = covering_satellites
            
            if is_valid:
                # Calculate total cost (sum of minimum costs for each location-time tuple)
                all_sats = set()
                for sats in coverage_details.values():
                    all_sats.update(sats)
                total_cost = sum(satellites[sat] for sat in all_sats)
                valid_solutions.append((satellite_set, total_cost, dict(coverage_details)))
    
    return sorted(valid_solutions, key=lambda x: x[1])  # Sort by total cost

def visualize_coverage(G, tuple_nodes, satellite_nodes):
    """
    Visualizes the complete coverage graph showing all possible coverages.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.bipartite_layout(G, tuple_nodes)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=tuple_nodes, 
                          node_color='lightblue', node_size=500,
                          label='Location-Time Pairs')
    nx.draw_networkx_nodes(G, pos, nodelist=satellite_nodes,
                          node_color='lightgreen', node_size=500,
                          label='Satellites')
    
    # Draw edges with different colors for different satellites
    colors = plt.cm.tab10(np.linspace(0, 1, len(satellite_nodes)))
    for idx, satellite in enumerate(satellite_nodes):
        edges = [(u, v) for (u, v) in G.edges() if satellite in (u, v)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, 
                             edge_color=[colors[idx]], alpha=0.5)
    
    # Labels
    labels = {**{node: f'{node[0]},{node[1]}' for node in tuple_nodes},
             **{node: node for node in satellite_nodes}}
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title("Complete Coverage Graph (All Possible Coverages)")
    plt.legend()
    plt.axis('off')
    return plt

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
else:
    print("\nMin cost satellite combination:")
    satellite_set, total_cost, coverage_details = valid_coverages[0]
    print(f"Satellites used: {satellite_set}")
    print(f"Total cost: {total_cost}")
    print("Coverage details:")
    for loc_time, satellites in coverage_details.items():
        print(f"  {loc_time} covered by:")
        for sat in satellites:
            print(f"    - {sat} at cost {satellite_nodes[sat]}")

print(f"\nSatellite costs: {satellite_nodes}")
print(f"Total number of valid solutions: {len(valid_coverages)}")
print(f"Time taken: {end_time - start_time} seconds")

if visualize:
    # Visualize the complete coverage
    plt = visualize_coverage(G, tuple_nodes, satellite_nodes)
    plt.savefig('complete_coverage_graph.png')
    plt.show()