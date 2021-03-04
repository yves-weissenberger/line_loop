import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import os, random
from math import sin, cos
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame, pygame.image, pygame.key
from pygame.locals import *
import time
import numpy as np
#import tensorflow as tf
import torch


#import matplotlib.pyplot as plt
textures = []
filter = 0
tris = []

yrot = 90.0
xpos = 0.0
ypos = 0.5
zpos = 0.0

lookupdown = 0.0
walkbias = 0.0
walkbiasangle = 0.0

LightAmbient  = [1.]*4#[ 0.9, 0.9, 0.9, 1.0]
LightDiffuse  = [1.]*4#[ 1.0, 1.0, 1.0, 1.0]

piover180 = 0.0174532925
window_size = (86,86)


#Task related stuff
hasrun = 10.
poke_ix = 0

possible_pos = [[-.95,.725,-0.02],
        [-.95,.65,-.3],[-.95,.65,.3],
    [-.95,.5,-.55],[-.95,.5,0.02],[-.95,.5,0.55],
        [-.95,.35,-.3],[-.95,.35,.3],
                [-.95,.3,-0.02]]

cpos = possible_pos[poke_ix]


##########################################
# Define neural network
##########################################


class Policy(nn.Module):
    def __init__(self,window_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(np.prod(window_size),28,bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.affine2 = nn.Linear(28, 6,bias=True)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy(window_size)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()



def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(gamma=.99):
    #gamma = 
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]




##Main task code

def resize(width, height):
    if height==0:
        height=1.0
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90, float(width)/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1, 1, 1, .5]);

    glLoadIdentity()

def init():
    global lookupdown, walkbias, walkbiasangle
    
    glEnable(GL_TEXTURE_2D)
    load_textures()
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    #glEnable(GL_LIGHTING)

    #glLightfv( GL_LIGHT0, GL_AMBIENT, LightAmbient )
    #glLightfv( GL_LIGHT0, GL_DIFFUSE, LightDiffuse )
    #glLightfv( GL_LIGHT0, GL_POSITION, LightPosition )


    #glLightfv( GL_LIGHT0, GL_SPECULAR, [1.,1.,1.,.5] )

    glLightfv( GL_LIGHT0, GL_AMBIENT,  [0.15]*3 + [1.])
    glLightfv( GL_LIGHT0, GL_DIFFUSE,  [0.15]*3 + [1.] )

    glLightfv( GL_LIGHT0, GL_POSITION, [0.,0.,0.,1.])
    #glLightfv( GL_LIGHT0,GL_SPOT_DIRECTION,[cos(.15),1.25,0.0,1.0])
    #glLightfv( GL_LIGHT0,GL_SPOT_CUTOFF,90.)

    #glLightfv( GL_LIGHT0,GL_EMISSION,1.)

    glEnable( GL_LIGHT0 )



    lookupdown    = 0.0
    walkbias      = 0.0
    walkbiasangle = 0.0
    glColor4f( 1.0, 1.0, 1.0, 0.5)


def load_textures():
    global textures
    
    textureSurface = pygame.image.load(os.path.join('/Users/yves/Desktop/nehe1-10/data','9pk.jpg'))
    textureData = pygame.image.tostring(textureSurface, "RGBX", 1)  #convert texturedata to string

    textureSurface2 = pygame.image.load(os.path.join('/Users/yves/Desktop/nehe1-10/data','pvc.jpg'))
    textureData2 = pygame.image.tostring(textureSurface2, "RGBX", 1)  #convert texturedata to string

    textures = [glGenTextures(1),glGenTextures(1)]     #define one texture
    
    glBindTexture(GL_TEXTURE_2D, textures[0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)#
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureSurface.get_width(), textureSurface.get_height(), 0,
                  GL_RGBA, GL_UNSIGNED_BYTE, textureData )
    
    glBindTexture(GL_TEXTURE_2D, textures[1])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)#
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureSurface2.get_width(), textureSurface2.get_height(), 0,
                  GL_RGBA, GL_UNSIGNED_BYTE, textureData2 )


def setup_world():
    global tris
    verts = 0
    tri = []
    
    f = open(os.path.join("/Users/yves/Desktop/nehe1-10/data", "world2.txt"))
    lines = f.readlines()
    f.close()
    for line in lines:
        vals = line.split()
        if len(vals) != 5:
            continue
        if vals[0] == '//':
            continue
        
        vertex = []
        for val in vals:
            vertex.append(float(val))
        tri.append(vertex)
        verts += 1
        if (verts == 3):
            tris.append(tri)
            tri = []
            verts = 0
    #print(len(tris))

def draw():
    global xpos, zpos, ypos, yrot, walkbias, lookupdown
    global textures, filter, tris, window_size,possible_pos, cpos, poke_ix, hasrun

    #hasrun = False
    xtrans = -xpos
    ztrans = -zpos
    ytrans = -ypos#-walkbias-0.25
    sceneroty=360.0-yrot

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glColor3f(1.0, 0.0, 0.0); #/* RED */



    pos = np.array([xpos,ypos,zpos])
    #print((np.abs(pos[1:] - cpos[1:]).sum()))
    if ((np.abs(pos[1:] - cpos[1:]).sum()<.3) and np.abs(pos[0]-cpos[0])<.2):
        poke_ix = (poke_ix + 1) % 8
        R = 10.
        print("RRRR")
    else:
        R = -.5

    cpos = possible_pos[poke_ix]



    glBegin(GL_POLYGON)

    radius = .1
    side_num = 20
    for vertex in range(side_num):
        angle  = float(vertex) * 2.0 * np.pi / side_num
        glVertex3f(cpos[0],cpos[1]+np.cos(angle)*radius, cpos[2]+np.sin(angle)*radius)
        #glVertex3f(np.cos(angle)*radius, .5+np.sin(angle)*radius,-.5)

    glEnd();



    #glClear( GL_COLOR_BUFFER_BIT)
    glColor([.1,1,1])

    glLoadIdentity()
    glRotatef( lookupdown, 1.0, 0.0 , 0.0 )
    glRotatef( sceneroty, 0.0, 1.0 , 0.0 )
    glTranslatef( xtrans, ytrans, ztrans )
    glBindTexture( GL_TEXTURE_2D, textures[1] )
    glShadeModel(GL_SMOOTH)

    #loop over all triangles in the images and draw them
    for tri in tris[:-2]:
        glBegin(GL_TRIANGLES)
        glNormal3f( 0.0, 1.0, 1.0)
        
        glTexCoord2f(tri[0][3], tri[0][4])  
        glVertex3f(tri[0][0], tri[0][1], tri[0][2])

        glTexCoord2f(tri[1][3], tri[1][4])
        glVertex3f(tri[1][0], tri[1][1], tri[1][2])

        glTexCoord2f(tri[2][3], tri[2][4])
        glVertex3f(tri[2][0], tri[2][1], tri[2][2])

        glEnd()


    glBindTexture( GL_TEXTURE_2D, textures[0] )
    for tri in tris[-2:]:
        glBegin(GL_TRIANGLES)
        glNormal3f( 0.0, 1.0, 1.0)
        
        glTexCoord2f(tri[0][3], tri[0][4])  
        glVertex3f(tri[0][0], tri[0][1], tri[0][2])

        glTexCoord2f(tri[1][3], tri[1][4])
        glVertex3f(tri[1][0], tri[1][1], tri[1][2])

        glTexCoord2f(tri[2][3], tri[2][4])
        glVertex3f(tri[2][0], tri[2][1], tri[2][2])

        glEnd()
   

    #glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
    dat = glReadPixels(0, 0, window_size[0],window_size[1], GL_RGB, GL_FLOAT)
    #print(dat)
    if hasrun<0:
        plt.figure()
        print(dat.shape)
        plt.imshow(dat[:,:,0].swapaxes(0,1).reshape(dat[:,:,0].shape,order='F').T,interpolation='None')
        plt.show()
        hasrun += 1
    return dat[:,:,0].swapaxes(0,1).reshape(dat[:,:,0].shape,order='F').T, R
def handle_keys(key):
    global xpos, zpos, ypos, yrot, lookupdown
    global piover180, walkbiasangle, walkbias
    
    if key==K_ESCAPE:
        return 0

    if key==0:
        xpos = np.clip(xpos -  sin( yrot * piover180 ) * 0.1,-.8,.8)
        zpos = np.clip(zpos - cos( yrot * piover180 ) * 0.1,-.8,.8)
        if ( walkbiasangle >= 359.0 ):
            walkbiasangle = 0.0
        else:
            walkbiasangle += 10.0
        walkbias = sin( walkbiasangle * piover180 ) / 20.0
    if key==1:
        xpos = np.clip(xpos +sin( yrot * piover180 ) * 0.05,-.8,.8)
        zpos = np.clip(zpos + cos( yrot * piover180 ) * 0.05,-.8,.8)
        if ( walkbiasangle <= 1.0 ):
            walkbiasangle = 359.0
        else:
            walkbiasangle -= 10.0
        walkbias = sin( walkbiasangle * piover180 ) / 20.0
    if key==2:
        ypos = np.clip(ypos + 0.01,0.2,0.8)
    if key==3:
        ypos = np.clip(ypos - 0.01,0.2,0.8)
    #if key==6:
    #    lookupdown += 2.
    #if key==7:
    #    lookupdown -= 2.

    if key==4:
        xpos = np.clip(xpos -  cos( yrot * piover180 ) * 0.1,-.8,.8)
        zpos = np.clip(zpos - sin( yrot * piover180 ) * 0.1,-.8,.8)
        if ( walkbiasangle >= 359.0 ):
            walkbiasangle = 0.0
        else:
            walkbiasangle -= 10.0
        walkbias = sin( walkbiasangle * piover180 ) / 20.0
    if key==5:
        xpos = np.clip(xpos + cos( yrot * piover180 ) * 0.05,-.8,.8)
        zpos = np.clip(zpos + sin( yrot * piover180 ) * 0.05,-.8,.8)
        if ( walkbiasangle <= 1.0 ):
            walkbiasangle = 359.0
        else:
            walkbiasangle += 10.0
        walkbias = sin( walkbiasangle * piover180 ) / 20.0

    #print(key)
    return 1

def main():

    global surface, window_size, xpos, ypos, zpos

    video_flags = OPENGL|DOUBLEBUF
    
    #window_size = (640,480)
    pygame.init()
    surface = pygame.display.set_mode(window_size, video_flags)
    pygame.key.set_repeat(1,10)

    #random.seed()
    resize(window_size[0],window_size[1])
    init()
    setup_world()

    frames = 0
    done = 0
    #clock = pygame.time.Clock()
    #clock.tick(150)

    ticks = pygame.time.get_ticks()
    st = time.time()
    totR = 0 
    all_actions = []
    while not done:
        #while 1:
        event = pygame.event.poll()
        #if event.type == NOEVENT:
        #    break
        if event.type == KEYDOWN:
            if event.key==K_UP:
                print(action)

        state,R = draw()
        totR += R
        #print(state.flatten().shape,state.shape)
        action = select_action(state.flatten()/255.)
        all_actions.append(action)
        handle_keys(action)
        policy.rewards.append(R)


        #print(xpos,ypos,zpos)
        #print("fps:  %d" % ((frames)/(time.time() - st)))

        #time.sleep(1)
        pygame.display.flip()
        frames += 1

        if np.remainder(frames,201)==0:
            print('action fraction {}'.format(np.array([all_actions.count(i) for i in np.unique(all_actions)])/len(all_actions)))
            finish_episode()

        if np.remainder(frames,201)==0:
            xpos = 0.
            ypos = .1
            zpos = 0.
            print("RESET")
            print(frames,totR,totR/frames,time.time() - st)

        #print(1/(time.time() - st))





if __name__ == '__main__': 
    main()
