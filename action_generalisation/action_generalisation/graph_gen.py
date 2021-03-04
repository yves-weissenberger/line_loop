import numpy as np

#cartesian product of two path graphs
def gen_path_graph(n):
    edges = [[i,i+1] for i in range(n-1)]
    edges = np.concatenate([edges,[np.flipud(i) for i in edges]])
    A = np.zeros([n,n])
    D = np.eye(n)
    for e in edges:
        A[e[0],e[1]] = 1
    D = np.eye(n)*A.sum(axis=1)
    L = D - A
    return A,D,L


def gen_path_plus(n):
    edges = [[i,i+1] for i in range(n-1)]
    edges = np.concatenate([edges,[np.flipud(i) for i in edges]])
    #edges = np.concatenate([edges,[[n-1,3],[3,n-1]]])
    edges = np.concatenate([edges,[[2,4],[4,2]]])
    A = np.zeros([n,n])
    D = np.eye(n)
    for e in edges:
        A[e[0],e[1]] = 1
    D = np.eye(n)*A.sum(axis=1)
    L = D - A
    return A,D,L

def gen_cycle_graph(n):
    edges = [[i,(i+1)%(n)] for i in range(n)]
    edges = np.concatenate([edges,[np.flipud(i) for i in edges]])
    A = np.zeros([n,n])
    D = np.eye(n)
    for e in edges:
        A[e[0],e[1]] = 1
    D = np.eye(n)*A.sum(axis=1)
    L = D - A
    return A,D,L


def gen_tree_graph(depth=3,max_b=3):
    
    edges = []
    ctr = 0
    for d in range(depth):
        n_branch = np.random.randint(0,max_b)
        
    
def simple_tree(n=8):
    edges = [[0,1],[1,2],[1,3],[2,4],[2,5],[3,6],[3,7]]
    edges = np.concatenate([edges,[np.flipud(i) for i in edges]])
    A = np.zeros([n,n])
    D = np.eye(n)
    for e in edges:
        A[e[0],e[1]] = 1
    D = np.eye(n)*A.sum(axis=1)
    L = D - A
    return A,D,L

def gen_star_graph(n=0):
    n=5
    edges = [[0,1],[0,2],[0,3],[0,4]]
    edges = np.concatenate([edges,[np.flipud(i) for i in edges]])
    A = np.zeros([n,n])
    D = np.eye(n)
    for e in edges:
        A[e[0],e[1]] = 1
    D = np.eye(n)*A.sum(axis=1)
    L = D - A
    return A,D,L

def gen_random_graph(gtype='ER',n=5,m=4,p=.3):
    if gtype=='ER':
        g = nx.random_graphs.erdos_renyi_graph(n,.7)
    elif gtype=='BB':
        g = nx.generators.barbell_graph(n,2)
    elif gtype=='FC':
        g = nx.complete_graph(n)
    elif gtype=='EMPTY':
        g = nx.empty_graph(n=n)
        g.add_edges_from([[1,0],[0,1]])
    elif gtype=='gnm':
        done = False
        while not done:
            g = nx.random_graphs.gnm_random_graph(5,4)
            done = nx.is_connected(g)
            
    elif gtype=='WS':
        g = nx.connected_watts_strogatz_graph(n,m,p=.3)
    elif gtype=='ROOM':
        g = nx.grid_2d_graph(n,m)
        
    elif gtype=='BI':
        done = False
        while not done:

            g = nx.bipartite.random_graph(n,m,p)
            done = nx.is_connected(g)

    A = nx.to_numpy_matrix(g)
    n = A.shape[0]
    D = np.eye(n)*A.sum(axis=1)
    L = D - A

    return A,D,L


def gen_outerplanar():
    n = 8
    edges = [[0,1],[1,2],[2,3],[2,0],[3,4],[3,0],[4,5],[5,6],[5,3],[6,7],[6,3],[7,3],[7,0]]
    edges = np.concatenate([edges,[np.flipud(i) for i in edges]])
    A = np.zeros([n,n])
    D = np.eye(n)
    for e in edges:
        A[e[0],e[1]] = 1
    D = np.eye(n)*A.sum(axis=1)
    L = D - A
    return A,D,L