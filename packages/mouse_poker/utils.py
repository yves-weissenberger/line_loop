import itertools

def get_empty_transition_dict(n_nodes=9,n_tasks=2):
    tmp1 = [str(i[1])+str(i[-2]) for i in list(itertools.combinations(range(n_nodes),2))]
    tmp1.extend([i[::-1] for i in tmp1])

    if n_tasks==0:
        tmp = tmp1
    else:
        tmp = []
        for tNr in range(1,1+n_tasks):
            tmp.extend([i+'_'+str(tNr) for i in tmp1])

    res_dct = {}
    for i in tmp:
        res_dct[i] = []
    return res_dct