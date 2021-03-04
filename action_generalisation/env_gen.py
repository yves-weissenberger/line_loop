import numpy as np
from skimage import measure


def generate_random_connected_maze(Nx,Ny,n_elim):
    """ Generate fully connected mazy by eliminating squares """
    M = np.ones([Nx,Ny])
    all_labels = measure.label(M)


    for _ in range(n_elim):
        Mdash = M.copy()
        Mdash[np.random.randint(0,Nx),np.random.randint(0,Ny)] = 0
        all_labels = measure.label(Mdash,connectivity=1)
        if len(np.unique(all_labels))<=2:
            M = Mdash
    return M


def generate_env_constrained(Mstart,protect,n_elim,verbose=False):

    all_labels = measure.label(Mstart)

    M = np.ones_like(Mstart)
    for ix in protect:
        M[ix[0],ix[1]] = Mstart[ix[0],ix[1]]   
    Nx,Ny = Mstart.shape
    for _ in range(n_elim):
        Mdash = M.copy()
        updX,updY = (np.random.randint(0,Nx),np.random.randint(0,Ny))
        
        if not np.any([np.all([updX,updY]==i) for i in protect]):
            if verbose: print((updX,updY),'no')
            Mdash[updX,updY] = 0
            all_labels = measure.label(Mdash,connectivity=1)
            if len(np.unique(all_labels))<=2:
                M = Mdash
        else:
            if verbose: print((updX,updY),'yes')

    return M
