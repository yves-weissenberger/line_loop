import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import re 
import sys
import time
import datetime


def get_active_inactive_rects(current_transitions,nStates=9):
    """ Function that gets active and inactive rectangles"""
    rects_active = []; rects_inactive = []

    for s in current_transitions:
        rects_active.append(pygame.draw.rect(screen,(255,0,0),rect_locs[s]+rect_size))
        
    for s in range(9):
        if s not in current_transitions:
            rects_inactive.append(pygame.draw.rect(screen,(255,255,255),rect_locs[s]+rect_size))
    return rects_active,rects_inactive


def select_graph(det=None,nNodes=9,nActions=2):
    """ """

    edges = [[0, 1],
             [0, 3],
             [1, 0],
             [1, 3],
             [2, 0],
             [2, 1],
             [3, 4],
             [3, 7],
             [4, 2],
             [4, 6],
             [5, 2],
             [5, 4],
             [6, 7],
             [6, 8],
             [7, 5],
             [7, 8],
             [8, 6],
             [8, 5]]

    available_transitions = np.zeros([nNodes,nActions],dtype='int')
    cntArr = np.zeros(nNodes,dtype='int')

    for fst,snd in edges:
        #edges.append([fst,snd])
        available_transitions[fst,cntArr[fst]] = snd
        cntArr[fst] += 1

    return edges,available_transitions

def select_layout():
    pass


ROOT = os.path.split(os.path.realpath("__file__"))[0]


if __name__ == "__main__":

    data_dir = os.path.join(ROOT,'data')
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    subject = input("Subject:")
    #print(subject)

    NOW = datetime.datetime.now().strftime('D_%Y_%m_%d_t_%H_%M_%S')

    data_file = os.path.join(data_dir,'_'.join([subject,NOW])+'.txt')
    print(data_file)


    ###############################################################################
    ### Initialise stuff for the task
    ###############################################################################

    nNodes = 9
    nActions = 2

    edges,available_transitions = select_graph()

    #locations of rectangles
    rect_locs = [[440,50],
                 [365,150],
                 [515,150],
                 [290,250],
                 [440,250],
                 [590,250],
                 [365,350],
                 [515,350],
                 [440,450]]

    #sizes of rectangles
    rect_size = [80,80]

    clock = pygame.time.Clock()
    FPS = 30
    pygame.init()
    font = pygame.font.Font(None,32)
    text = font.render("Score: 0",True,(120,0,0))

    size = (960, 640)
    screen = pygame.display.set_mode(size)


    BLACK = [0,0,0]
    screen.fill(BLACK)
    pygame.display.update()


    playing = True

    state = start_state = np.random.randint(0,9)
    current_transitions = available_transitions[state]

    SCORE = 0

    rewarded_states = [np.random.randint(0,9)]

    RS_PROB = .01
    TELEPORT_PROB = .5

    rects_active,rects_inactive = get_active_inactive_rects(current_transitions,nNodes)


    ###############################################################################
    ### Run the task
    ###############################################################################
    while playing:
        
        for event in pygame.event.get(): 
            
            if event.type == pygame.QUIT:
                playing=False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                collision = any([r_.collidepoint(pos) for r_ in rects_active])
                
                if collision:
                    screen.fill(BLACK)
                    print(rewarded_states,state)
                    for kk,r_ in enumerate(rects_active):
                        if r_.collidepoint(pos):
                            state = available_transitions[state,kk]
                            
                            #with small probability teleport
                            if state not in rewarded_states:
                                if np.random.rand()<TELEPORT_PROB:
                                    print("RANDOM TRANSITION")
                                    state = np.random.permutation([i for i in range(9) if i not in rewarded_states])[0]#np.random.randint(0,9)

                            current_transitions = available_transitions[state]



                            rects_active,rects_inactive = get_active_inactive_rects(current_transitions,nNodes)


                            if state in rewarded_states:
                                pygame.draw.rect(screen,(0,0,255),[r_.x,r_.y,r_.width,r_.height])
                                SCORE += 5
                                text = font.render('Score:' + str(SCORE),True,(120,0,0))
                                pygame.display.flip()

                            time.sleep(0.1)
                    #switch reward location probabilistically
                    if np.random.rand()<RS_PROB:
                        print("REW SWITCH")
                        rewarded_states = [np.random.randint(0,9)]
                        
                        
                
                    
        screen.blit(text,(120,150))
        clock.tick(FPS)
        pygame.display.flip()

    pygame.quit()