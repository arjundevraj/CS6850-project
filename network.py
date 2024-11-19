import networkx as nx
import matplotlib.pyplot as plt
import random

def create_satellite_bipartite_graph(locations, timesteps, satellites, min_cost=0, max_cost=10):
    """
    Creates a bipartite graph between (location,timestep) tuples and satellites.
    
    Args:
        locations (int): Number of locations
        timesteps (int): Number of time steps
        satellites (int): Number of satellites
    
    Returns:
        G: NetworkX graph
        tuple_nodes: List of (location, timestep) tuples
        satellite_nodes: List of satellite nodes
    """
    G = nx.Graph()
    edge_costs = {}
    
    # Create (location, timestep) tuple nodes
    tuple_nodes = []
    for l in range(locations):
        for t in range(timesteps):
            node = (f'L{l}', f'T{t}')
            tuple_nodes.append(node)
            G.add_node(node, bipartite=0)  # First set of nodes in bipartite graph
            
    # Create satellite nodes
    satellite_nodes = [f'S{s}' for s in range(satellites)]
    G.add_nodes_from(satellite_nodes, bipartite=1)  # Second set of nodes
    
    # Add random edges (each satellite can cover some location-time pairs)
    for s in satellite_nodes:
        # Each satellite covers a random number of location-time pairs
        num_coverages = random.randint(1, len(tuple_nodes))
        covered_tuples = random.sample(tuple_nodes, num_coverages)
        for tuple_node in covered_tuples:
            cost = random.randint(min_cost, max_cost)
            G.add_edge(tuple_node, s)
            edge_costs[(tuple_node, s)] = cost

    return G, tuple_nodes, satellite_nodes, edge_costs

def visualize_bipartite_graph(G, tuple_nodes, satellite_nodes, edge_costs):
    """
    Visualizes the bipartite graph with edge costs.
    
    Args:
        G: NetworkX graph
        tuple_nodes: List of (location, timestep) tuples
        satellite_nodes: List of satellite nodes
        edge_costs: Dictionary of edge costs
    """
    plt.figure(figsize=(12, 8))
    
    # Create bipartite layout
    pos = nx.bipartite_layout(G, tuple_nodes)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=tuple_nodes, 
                          node_color='lightblue', node_size=500,
                          label='Location-Time Pairs')
    nx.draw_networkx_nodes(G, pos, nodelist=satellite_nodes,
                          node_color='lightgreen', node_size=500,
                          label='Satellites')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Create labels
    tuple_labels = {node: f'{node[0]},{node[1]}' for node in tuple_nodes}
    satellite_labels = {node: node for node in satellite_nodes}
    labels = {**tuple_labels, **satellite_labels}
    
    # Create edge labels with costs
    edge_labels = {(u, v): f'cost={edge_costs.get((u, v)) or edge_costs.get((v, u))}' 
                  for (u, v) in G.edges()}
    
    nx.draw_networkx_labels(G, pos, labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title("Satellite Coverage Bipartite Graph with Costs")
    plt.legend()
    plt.axis('off')
    
    return plt

        
# Example usage
locations = 3
timesteps = 5
satellites = 3
min_cost = 0
max_cost = 10

# Create and visualize the graph
G, tuple_nodes, satellite_nodes, edge_costs = create_satellite_bipartite_graph(
    locations, timesteps, satellites, min_cost, max_cost
)

print("Location-Time Tuples:")
for node in sorted(tuple_nodes):
    print(node)

print("\nSatellites:")
for node in satellite_nodes:
    print(node)

print("\nCoverage with Costs:")
for (u, v) in G.edges():
    cost = edge_costs.get((u, v)) or edge_costs.get((v, u))
    print(f"{u} covered by {v} with cost {cost}")

plt = visualize_bipartite_graph(G, tuple_nodes, satellite_nodes, edge_costs)
plt.show()