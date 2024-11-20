import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations, chain
from collections import defaultdict

def create_satellite_bipartite_graph(locations, timesteps, satellites, min_cost=1, max_cost=10):
    """Previous function with increased coverage probability"""
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
    satellite_nodes = [f'S{s}' for s in range(satellites)]
    G.add_nodes_from(satellite_nodes, bipartite=1)
    
    # Add random edges with costs - increased probability for multiple coverage
    for s in satellite_nodes:
        # Each satellite covers more location-time pairs
        num_coverages = random.randint(len(tuple_nodes) // 2, len(tuple_nodes))
        covered_tuples = random.sample(tuple_nodes, num_coverages)
        for tuple_node in covered_tuples:
            cost = random.randint(min_cost, max_cost)
            G.add_edge(tuple_node, s)
            edge_costs[(tuple_node, s)] = cost
    
    return G, tuple_nodes, satellite_nodes, edge_costs

def get_coverage_map(G, tuple_nodes, satellite_nodes, edge_costs):
    """
    Creates a mapping of each location-time tuple to all satellites that can cover it.
    
    Returns:
        dict: Mapping of tuple -> list of (satellite, cost) pairs
    """
    coverage_map = defaultdict(list)
    for tuple_node in tuple_nodes:
        for satellite in G.neighbors(tuple_node):
            cost = edge_costs.get((tuple_node, satellite)) or edge_costs.get((satellite, tuple_node))
            coverage_map[tuple_node].append((satellite, cost))
    return coverage_map

def find_all_valid_coverages(G, tuple_nodes, satellite_nodes, edge_costs):
    """
    Find all valid combinations of satellites that provide full coverage.
    Each location-time tuple can be covered by multiple satellites.
    
    Returns:
        list: List of (satellite_set, total_cost, coverage_details) tuples
    """
    coverage_map = get_coverage_map(G, tuple_nodes, satellite_nodes, edge_costs)
    valid_solutions = []
    
    # Check if each location-time tuple has at least one satellite covering it
    for tuple_node, coverages in coverage_map.items():
        if not coverages:
            print(f"Warning: {tuple_node} has no satellite coverage!")
            return []

    # Try all possible combinations of satellites
    for r in range(1, len(satellite_nodes) + 1):
        for satellite_subset in combinations(satellite_nodes, r):
            satellite_set = set(satellite_subset)
            
            # Check if this combination provides full coverage
            coverage_details = defaultdict(list)
            is_valid = True
            
            # Check each location-time tuple
            for tuple_node in tuple_nodes:
                # Get all satellites from our subset that can cover this tuple
                covering_satellites = [(sat, cost) for sat, cost in coverage_map[tuple_node] 
                                    if sat in satellite_set]
                
                if not covering_satellites:
                    is_valid = False
                    break
                    
                coverage_details[tuple_node] = covering_satellites
            
            if is_valid:
                # Calculate total cost (sum of minimum costs for each location-time tuple)
                total_cost = sum(min(cost for _, cost in satellites) 
                               for satellites in coverage_details.values())
                valid_solutions.append((satellite_set, total_cost, dict(coverage_details)))
    
    return sorted(valid_solutions, key=lambda x: x[1])  # Sort by total cost

# Example usage
locations = 2
timesteps = 2
satellites = 3
min_cost = 1
max_cost = 10

# Create graph
G, tuple_nodes, satellite_nodes, edge_costs = create_satellite_bipartite_graph(
    locations, timesteps, satellites, min_cost, max_cost
)

# Find all valid coverages
valid_coverages = find_all_valid_coverages(G, tuple_nodes, satellite_nodes, edge_costs)

# Print results
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

def visualize_coverage(G, tuple_nodes, satellite_nodes, edge_costs):
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
    
    # Edge labels
    edge_labels = {(u, v): f'cost={edge_costs.get((u, v)) or edge_costs.get((v, u))}' 
                  for (u, v) in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("Complete Coverage Graph (All Possible Coverages)")
    plt.legend()
    plt.axis('off')
    return plt

# Visualize the complete coverage
import numpy as np
plt = visualize_coverage(G, tuple_nodes, satellite_nodes, edge_costs)
plt.savefig('complete_coverage_graph.png')
plt.show()