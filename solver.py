from gurobipy import *
from scipy.stats import bernoulli

def weighted_set_cover_lp_relaxation(G, tuple_nodes, satellite_nodes, k):
    """
    Solve the LP relaxation of the set cover problem, with k sampling iterations
    
    Returns:
        tuple: (satellite_set, total_cost)
    """
    m = Model("weighted_set_cover_ilp")
    
    # Create a mapping of each location-time tuple to all satellites that can cover it
    coverage_map = {}
    for sat in satellite_nodes:
        coverage_map[sat] = set(G.neighbors(sat))
    
    # Initialize the set of all location-time tuples
    U = set(tuple_nodes)
    
    # Initialize an empty set to store the selected satellites
    satellite_set = set()
    
    # Create a binary variable for each satellite
    x = {}
    for sat in satellite_nodes:
        x[sat] = m.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f"x_{sat}")
    
    # Set objective function
    m.setObjective(quicksum(satellite_nodes[sat] * x[sat] for sat in satellite_nodes), GRB.MINIMIZE)
    
    # Add constraints
    for tuple_node in tuple_nodes:
        if G.degree(tuple_node) == 0:
            continue
        m.addConstr(quicksum(x[sat] for sat in coverage_map if tuple_node in coverage_map[sat]) >= 1)
    
    m.optimize()
    
    for i in range(k):
        for sat in satellite_nodes:
            sample = bernoulli.rvs(x[sat].x)
            if sample == 1:
                satellite_set.add(sat)
    
    total_cost = sum(satellite_nodes[sat] for sat in satellite_set)
    
    return satellite_set, total_cost


def weighted_set_cover_ilp(G, tuple_nodes, satellite_nodes):
    """
    Solve the LP relaxation of the set cover problem, with k sampling iterations
    
    Returns:
        tuple: (satellite_set, total_cost)
    """
    m = Model("weighted_set_cover_ilp")
    
    # Create a mapping of each location-time tuple to all satellites that can cover it
    coverage_map = {}
    for sat in satellite_nodes:
        coverage_map[sat] = set(G.neighbors(sat))
    
    # Initialize the set of all location-time tuples
    U = set(tuple_nodes)
    
    # Initialize an empty set to store the selected satellites
    satellite_set = set()
    
    # Create a binary variable for each satellite
    x = {}
    for sat in satellite_nodes:
        x[sat] = m.addVar(vtype=GRB.BINARY, name=f"x_{sat}")
    
    # Set objective function
    m.setObjective(quicksum(satellite_nodes[sat] * x[sat] for sat in satellite_nodes), GRB.MINIMIZE)
    
    # Add constraints
    for tuple_node in tuple_nodes:
        if G.degree(tuple_node) == 0:
            continue
        m.addConstr(quicksum(x[sat] for sat in coverage_map if tuple_node in coverage_map[sat]) >= 1)
    
    m.optimize()
    
    for sat in satellite_nodes:
        if x[sat].x == 1:
            satellite_set.add(sat)
    
    total_cost = sum(satellite_nodes[sat] for sat in satellite_set)
    
    return satellite_set, total_cost

def weighted_set_cover_ilp_tradeoff(G, tuple_nodes, satellite_nodes, l):
    """
    Solve the LP relaxation of the set cover problem, with k sampling iterations
    
    Returns:
        tuple: (satellite_set, total_cost)
    """
    m = Model("weighted_set_cover_ilp")
    
    # Create a mapping of each location-time tuple to all satellites that can cover it
    coverage_map = {}
    for sat in satellite_nodes:
        coverage_map[sat] = set(G.neighbors(sat))
    
    # Initialize the set of all location-time tuples
    U = set(tuple_nodes)
    
    # Initialize an empty set to store the selected satellites
    satellite_set = set()
    
    # Create a binary variable for each satellite
    x = {}
    for sat in satellite_nodes:
        x[sat] = m.addVar(vtype=GRB.BINARY, name=f"x_{sat}")
    
    y = {}
    q = {}
    M = len(satellite_nodes)
    for tuple_node in tuple_nodes:
        if G.degree(tuple_node) == 0:
            continue
        y[tuple_node] = m.addVar(vtype=GRB.INTEGER, name=f"y_{tuple_node}")
        m.addConstr(y[tuple_node] == quicksum(x[sat] for sat in coverage_map if tuple_node in coverage_map[sat]))
        q[tuple_node] = m.addVar(vtype=GRB.BINARY, name=f"q_{tuple_node}")
        # m.addConstr(y[tuple_node] >= q[tuple_node])
        # m.addConstr(y[tuple_node] <= M * q[tuple_node])
        m.addConstr(y[tuple_node] <= M * (1 - q[tuple_node]))
        m.addConstr(y[tuple_node] >= 1 - q[tuple_node])
    
    # Set objective function
    m.setObjective(quicksum(satellite_nodes[sat] * x[sat] for sat in satellite_nodes) + l * quicksum(q[tuple_node] for tuple_node in q), GRB.MINIMIZE)

    m.optimize()
    
    for sat in satellite_nodes:
        if x[sat].x == 1:
            satellite_set.add(sat)
    
    total_cost = sum(satellite_nodes[sat] for sat in satellite_set)

    assert total_cost >= 0
    
    return satellite_set, total_cost