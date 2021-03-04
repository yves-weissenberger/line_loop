import os
loc  = os.path.split(os.path.abspath(__file__))[0]
def load_graphs():
    print(loc)
    with open(os.path.join(loc,'regular_graphs_9_nodes_degree_2'),'r') as f:
        graphs = f.readlines()
    return graphs