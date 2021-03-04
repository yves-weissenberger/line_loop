import numpy as np


def reconstruct_graph_from_pokes(dat_dict):
    
    """ Function taskes in the dictionary of pokes and
        returns the edges of the graph defined by the sequence
        of observed pokes
    """
    print("WARNING CURRENTLY BUGGY BECAUSE OF TELEPORTS \nuse get_valid_edges instead")
    nEntries = len(dat_dict['random'])
    edges = []
    for i in range(nEntries):
        if not dat_dict['random'][i]:
            edges.append((dat_dict['port'][i][0],dat_dict['port'][i][1][0]))
            edges.append((dat_dict['port'][i][0],dat_dict['port'][i][1][1]))
    return edges


def get_valid_edges(dat_dict):
    edges = []
    for ctr,i in enumerate(dat_dict['port']):
        #if None not in i[1]:
        for j in i[1]:
            if j is not None and not dat_dict['random'][ctr+1]:
                if [i[0],j] not in edges: edges.append([i[0],j])
    return edges