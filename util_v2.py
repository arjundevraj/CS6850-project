import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Linux didn't like something you can remove this for mac
import matplotlib.pyplot as plt
import random
from itertools import combinations
from collections import defaultdict
import numpy as np
import math
from scipy.stats import bernoulli

def create_satellite_bipartite_graph(locations, timesteps, satellites, coverage_prob, min_cost=1, max_cost=10, class_ratios=[0.2, 0.3, 0.5], class_coverages=[0.8, 0.4, 0.1]):
    """Previous function with increased coverage probability"""
    assert coverage_prob >= 0 and coverage_prob <= 1, "Coverage probability must be between 0 and 1"
    G = nx.Graph()
    

    # switch order to calculate the coverage based on a different distribution followed by the coverage probability
    # then calculate the cost for each node given the degree
    # Create (location, timestep) tuple nodes
    tuple_nodes = []
    for l in range(locations):
        for t in range(timesteps):
            node = (f'L{l}', f'T{t}')
            tuple_nodes.append(node)
            G.add_node(node, bipartite=0)


    # calculate # of satellites per class (aka how many big players, medium players, small players)
    satellites_per_class = [math.ceil(satellites * ratio) for ratio in class_ratios]
    


    # Create satellite nodes
    current_satellite = 0
    satellite_nodes = {}
    simpler_satellite_nodes = {}
    for class_idx, num_sats in enumerate(satellites_per_class):
        class_name = f"Class-{class_idx+1}"
        base_coverage = class_coverages[class_idx] # coverage probability for this class of satellite provider
        
        for _ in range(num_sats):
            node = f'S{current_satellite}'
            simpler_satellite_nodes[node] = 0
            satellite_nodes[node] = {
                'class': class_name,
                'base_coverage': base_coverage,
                'cost': None  # Will be set later based on # of covered tuples (edges)
            }
            G.add_node(node, bipartite=1, cost=0)
            current_satellite += 1

    # Adding edges based on class coverage probabilities
    for s in satellite_nodes:
        base_coverage = satellite_nodes[s]['base_coverage']

        coverage_variation = 0.1
        # essentially, we are adding some noise to the base coverage probability which represents the liklihood of a satellite covering a location-time pair
        actual_coverage = np.clip(base_coverage + np.random.normal(0, coverage_variation), 0, 1)

        # scaling by overall coverage probability
        final_coverage = actual_coverage * coverage_prob

        # coverage mask (number of location-time pairs covered by this satellite)
        coverage = np.random.random(len(tuple_nodes)) < final_coverage

        for tuple_node, covered in zip(tuple_nodes, coverage):
            if covered:
                G.add_edge(tuple_node, s)
    
    # calculate cost for each satellite based on the number of location-time pairs it covers

    max_possible_degree = locations * timesteps

    for s in satellite_nodes:
        degree = G.degree(s)
        base_coverage = satellite_nodes[s]['base_coverage']

        # satellites with higher base coverage cost more, even with same degree of a satellite 
        # (more expensive plans come from companies with higher base coverage) so, if the prob was 0.3, fact is 1.3
        class_cost_factor = 1 + base_coverage

        # combining degree and class cost factor to get the final cost
        normalized_degree = degree / max_possible_degree
        cost = int(min_cost + (normalized_degree * class_cost_factor) * (max_cost - min_cost))

        satellite_nodes[s]['cost'] = cost
        simpler_satellite_nodes[s] = cost
        G.nodes[s]['cost'] = cost


    
    return G, tuple_nodes, simpler_satellite_nodes#satellite_nodes

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

def brute_force_algorithm(G, feasible_tuple_nodes, satellites):
    best_solution = None 
    best_cost = float('inf')
    for r in range(1, len(satellites) + 1):
        for satellite_subset in combinations(satellites, r):
            satellite_set = set(satellite_subset)
            covered_tuples = set()
            for satellite in satellite_set:
                covered_tuples.update(G.neighbors(satellite))
        
            if len(covered_tuples) == len(feasible_tuple_nodes):
                total_cost = sum(satellites[satellite] for satellite in satellite_set)
                if total_cost < best_cost:
                    best_solution = satellite_set
                    best_cost = total_cost
    return best_solution, best_cost

def greedy_degree_based_algorithm(G, feasible_tuple_nodes, satellites):
    """
    Find the optimal combination of satellites that provides full coverage.
    Each location-time tuple can be covered by multiple satellites.
    
    Returns:
        tuple: (satellite_set, total_cost, coverage_details)
    """
    # Create a mapping of each location-time tuple to all satellites that can cover it
    coverage_map = {}
    for sat in satellites:
        coverage_map[sat] = set(G.neighbors(sat))
    
    coverage_map_sorted = {k: v for k, v in sorted(coverage_map.items(), key=lambda item: len(item[1]), reverse=True)}
    
    # Initialize the set of all location-time tuples
    U = set(feasible_tuple_nodes)
    
    # Initialize an empty set to store the selected satellites
    satellite_set = set()

    for sat, tuples in coverage_map_sorted.items():
        if len(U) == 0:
            break
        covered = U.intersection(tuples)
        if len(covered) > 0:
            satellite_set.add(sat)
            U -= covered
    
    # Calculate the total cost
    total_cost = sum(satellites[satellite] for satellite in satellite_set)
    
    return satellite_set, total_cost

def greedy_cost_based_algorithm(G, feasible_tuple_nodes, satellites):
    """
    Find the optimal combination of satellites that provides full coverage.
    Each location-time tuple can be covered by multiple satellites.
    
    Returns:
        tuple: (satellite_set, total_cost, coverage_details)
    """
    # Create a mapping of each location-time tuple to all satellites that can cover it
    coverage_map = {}
    for sat in satellites:
        coverage_map[sat] = set(G.neighbors(sat))
    
    # sort coverage_map by the value of the satellite in satellites dict
    coverage_map_sorted = {k: v for k, v in sorted(coverage_map.items(), key=lambda item: satellites[item[0]])}
    
    # Initialize the set of all location-time tuples
    U = set(feasible_tuple_nodes)
    
    # Initialize an empty set to store the selected satellites
    satellite_set = set()

    for sat, tuples in coverage_map_sorted.items():
        if len(U) == 0:
            break
        covered = U.intersection(tuples)
        if len(covered) > 0:
            satellite_set.add(sat)
            U -= covered
    
    # Calculate the total cost
    total_cost = sum(satellites[satellite] for satellite in satellite_set)
    
    return satellite_set, total_cost

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