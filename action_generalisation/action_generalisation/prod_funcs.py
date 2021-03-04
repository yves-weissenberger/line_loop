import numpy as np

#generate hypercube

def graph_cart_product(A,B=None):
    if B is None: B=A.copy()
    
    n,m = A.shape[0],B.shape[0]
    return np.kron(A,np.eye(m)) + np.kron(np.eye(n),B)


#generate hypercube

def strong_product(A,B=None):
    if B is None: B=A.copy()
    
    n,m = A.shape[0],B.shape[0]
    return np.kron(A+ np.eye(n),np.eye(m)+B) - np.eye(n+m)

def eig_sort(M):
    eigenval,v = np.linalg.eig(M)
    v = v[:,np.argsort((eigenval))[::-1]]
    eigenval = eigenval[np.argsort((eigenval))[::-1]]
    return eigenval,v