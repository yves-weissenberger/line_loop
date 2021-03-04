import numpy as np
import networkx as nx

poke_pos = np.array([[8,10],
                    [7,8],[9,8],
                    [6,6],[8,6],[10,6],
                    [7,4],[9,4],
                    [8,2]])


poke_posN = poke_pos.copy()
poke_posN[:,0] -= 8
poke_posN[:,1] -= 6 


def draw_on_physical_layout(gw,pos):
    nx.draw(gw.G,
            pos=pos,#[np.random.permutation(range(9))],
            connectionstyle='arc3, rad=0.2',
            with_labels=True)