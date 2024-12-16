import networkx as nx
import numpy as np
import matplotlib
import util_v2 
import util
matplotlib.use('Agg')  # Set backend
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def analyze_coverage_distributions(G_dict):
    """
    Analyzes coverage patterns for different distribution types
    """
    results = {}
    
    for dist_name, G in G_dict.items():
        # Get satellite nodes (bipartite=1)
        satellite_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
        
        # Calculate degree (coverage) for each satellite
        degrees = [G.degree(s) for s in satellite_nodes]
        
        # Store statistics
        results[dist_name] = {
            'degrees': sorted(degrees, reverse=True),
            'mean': np.mean(degrees),
            'median': np.median(degrees),
            'std': np.std(degrees),
            'max': max(degrees),
            'min': min(degrees)
        }
    
    return results

def plot_coverage_comparison(graphs, save_path='coverage_comparison.png'):
    """
    Creates visualization comparing different coverage distributions
    """
    # Set up the plot style
    # plt.style.use('seaborn')
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig = plt.figure(figsize=(15, 12))
    
    # Analyze all distributions
    distributions = {
        'Bernoulli': graphs['bernoulli'],
        'Power Law': graphs['power_law'],
        'Lognormal': graphs['lognormal'],
        'Ours': graphs['Ours']
    }
    
    results = analyze_coverage_distributions(distributions)

    degrees_data = [stats['degrees'] for stats in results.values()]

    # only show the violin plot
    ax1 = fig.add_subplot(111)
    sns.violinplot(data=degrees_data, ax=ax1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(left=False, bottom=False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title('Coverage Density Distribution')
    ax1.set_ylabel('Number of Locations Covered')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



num_satellites_list = 10
num_timesteps_list = 3
num_locations_list = 6
COVERAGE_PROB = 0.5


# visualize difference in models
G_v2, tuple_nodes_v2, satellite_nodes_v2 = util_v2.create_satellite_bipartite_graph(num_locations_list, num_timesteps_list, num_satellites_list, COVERAGE_PROB)

v2_plt = util_v2.visualize_coverage(G_v2, tuple_nodes_v2, satellite_nodes_v2)
plt.savefig('v2.png')

G_bernoulli, tuple_nodes_bernoulli, satellite_nodes_bernoulli = util.create_satellite_bipartite_graph(num_locations_list, num_timesteps_list, num_satellites_list, COVERAGE_PROB, prob_type='bernoulli')

bernoulli_plt = util.visualize_coverage(G_bernoulli, tuple_nodes_bernoulli, satellite_nodes_bernoulli)
plt.savefig('bernoulli.png')

G_power_law, tuple_nodes_power_law, satellite_nodes_power_law = util.create_satellite_bipartite_graph(num_locations_list, num_timesteps_list, num_satellites_list, COVERAGE_PROB, prob_type='power_law')

power_law_plt = util.visualize_coverage(G_power_law, tuple_nodes_power_law, satellite_nodes_power_law)
plt.savefig('power_law.png')

G_lognormal, tuple_nodes_lognormal, satellite_nodes_lognormal = util.create_satellite_bipartite_graph(num_locations_list, num_timesteps_list, num_satellites_list, COVERAGE_PROB, prob_type='lognormal')

lognormal_plt = util.visualize_coverage(G_lognormal, tuple_nodes_lognormal, satellite_nodes_lognormal)
plt.savefig('lognormal.png')

graphs = {
    'bernoulli': G_bernoulli,
    'power_law': G_power_law,
    'lognormal': G_lognormal,
    'Ours': G_v2
}

plot_coverage_comparison(graphs)