import numpy as np
from itertools import count
from numpy import sin, cos
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame, pygame.image, pygame.key
from pygame.locals import *
import time
import sys, os, re


#Imports for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim


class poke_env(object):


    def __init__(self,window_size = (400,200),poke_ix = 0,fov=90.,dist_thresh=.5,human_mode=False,move_gain=0.25):

        """ Simple class that implements the poking task"""

        self.dist_thresh = dist_thresh
        self.window_size = window_size
        self.move_gain = move_gain
        self.possible_pos = [[-.95,.725,-0.02],
                        [-.95,.65,-.3],[-.95,.65,.3],
                    [-.95,.5,-.55],[-.95,.5,0.02],[-.95,.5,0.55],
                        [-.95,.35,-.3],[-.95,.35,.3],
                                [-.95,.3,-0.02]]

        self.piover180 = np.pi/180.

        self.yrot = 90.0
        self.xpos = 0.0
        self.ypos = 0.5
        self.zpos = 0.0

        self.poke_ix = poke_ix
        self.cpos = self.possible_pos[poke_ix]
        self.fov = fov
        self.width, self.height = window_size

        self.video_flags = OPENGL|DOUBLEBUF

        pygame.init()
        surface = pygame.display.set_mode(self.window_size, self.video_flags)
        pygame.key.set_repeat(1,10)


        self.action_space = {'n': 4}

        self.lookupdown    = 0.0
        self.walkbias      = 0.0
        self.walkbiasangle = 0.0

        self.resize()
        self.init_opengl()

        self.setup_world()
    



    def handle_keys(self,key):
        

        if key==0:
            self.xpos = np.clip(self.xpos -  sin( self.yrot * self.piover180 ) * self.move_gain,-.8,.8)
            self.zpos = np.clip(self.zpos - cos( self.yrot * self.piover180 ) * self.move_gain,-.8,.8)
            if ( self.walkbiasangle >= 359.0 ):
                self.walkbiasangle = 0.0
            else:
                self.walkbiasangle += 10.0
            self.walkbias = sin( self.walkbiasangle * self.piover180 ) / 20.0
        if key==1:
            self.xpos = np.clip(self.xpos +sin( self.yrot * self.piover180 ) * self.move_gain,-.8,.8)
            self.zpos = np.clip(self.zpos + cos( self.yrot * self.piover180 ) * self.move_gain,-.8,.8)
            if ( self.walkbiasangle <= 1.0 ):
                self.walkbiasangle = 359.0
            else:
                self.walkbiasangle -= 10.0
            self.walkbias = sin( self.walkbiasangle * self.piover180 ) / 20.0
        if key==2:
            self.ypos = np.clip(self.ypos + 0.01,0.2,0.8)
        if key==3:
            self.ypos = np.clip(self.ypos - 0.01,0.2,0.8)


        if key==4:
            self.xpos = np.clip(self.xpos -  cos( self.yrot * self.piover180 ) * 0.2,-.8,.8)
            self.zpos = np.clip(self.zpos - sin( self.yrot * self.piover180 ) * 0.2,-.8,.8)
            if ( self.walkbiasangle >= 359.0 ):
                self.walkbiasangle = 0.0
            else:
                self.walkbiasangle -= 10.0
            self.walkbias = sin( self.walkbiasangle * self.piover180 ) / 20.0
        if key==5:
            self.xpos = np.clip(self.xpos + cos( self.yrot * self.piover180 ) * 0.2,-.8,.8)
            self.zpos = np.clip(self.zpos + sin( self.yrot * self.piover180 ) * 0.2,-.8,.8)
            if ( self.walkbiasangle <= 1.0 ):
                self.walkbiasangle = 359.0
            else:
                self.walkbiasangle += 10.0
            self.walkbias = sin( self.walkbiasangle * self.piover180 ) / 20.0

        #print(key)
        return 1

    def init_opengl(self):
        
        glEnable(GL_TEXTURE_2D)
        self.load_textures()
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



        glColor4f( 1.0, 1.0, 1.0, 0.5)


    def draw(self):

        self.lookupdown = 0.
        #hasrun = False
        xtrans = -self.xpos
        ztrans = -self.zpos
        ytrans = -self.ypos
        sceneroty=360.0-self.yrot

        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glColor3f(1.0, 0.0, 0.0); #/* RED */



        pos = np.array([self.xpos,self.ypos,self.zpos])
        #print((np.abs(pos[1:] - cpos[1:]).sum()))
        if ((np.abs(pos[1:] - self.cpos[1:]).sum()<self.dist_thresh) and np.abs(pos[0]-self.cpos[0])<.2):
            self.poke_ix = (self.poke_ix + 1) % 8
            R = 1.
        else:
            R = -.05

        self.cpos = self.possible_pos[self.poke_ix]



        glBegin(GL_POLYGON)

        radius = .1
        side_num = 20
        for vertex in range(side_num):
            angle  = float(vertex) * 2.0 * np.pi / side_num
            glVertex3f(self.cpos[0],self.cpos[1]+np.cos(angle)*radius, self.cpos[2]+np.sin(angle)*radius)
            #glVertex3f(np.cos(angle)*radius, .5+np.sin(angle)*radius,-.5)

        glEnd();



        #glClear( GL_COLOR_BUFFER_BIT)
        glColor([.1,1,1])

        glLoadIdentity()
        glRotatef( self.lookupdown, 1.0, 0.0 , 0.0 )
        glRotatef( sceneroty, 0.0, 1.0 , 0.0 )
        glTranslatef( xtrans, ytrans, ztrans )
        glBindTexture( GL_TEXTURE_2D, self.textures[1] )
        glShadeModel(GL_SMOOTH)

        #loop over all triangles in the images and draw them
        for tri in self.tris[:-2]:
            glBegin(GL_TRIANGLES)
            glNormal3f( 0.0, 1.0, 1.0)
            
            glTexCoord2f(tri[0][3], tri[0][4])  
            glVertex3f(tri[0][0], tri[0][1], tri[0][2])

            glTexCoord2f(tri[1][3], tri[1][4])
            glVertex3f(tri[1][0], tri[1][1], tri[1][2])

            glTexCoord2f(tri[2][3], tri[2][4])
            glVertex3f(tri[2][0], tri[2][1], tri[2][2])

            glEnd()


        glBindTexture( GL_TEXTURE_2D, self.textures[0] )
        for tri in self.tris[-2:]:
            glBegin(GL_TRIANGLES)
            glNormal3f( 0.0, 1.0, 1.0)
            
            glTexCoord2f(tri[0][3], tri[0][4])  
            glVertex3f(tri[0][0], tri[0][1], tri[0][2])

            glTexCoord2f(tri[1][3], tri[1][4])
            glVertex3f(tri[1][0], tri[1][1], tri[1][2])

            glTexCoord2f(tri[2][3], tri[2][4])
            glVertex3f(tri[2][0], tri[2][1], tri[2][2])

            glEnd()
       

        dat = glReadPixels(0, 0, self.width,self.height, GL_RGB, GL_FLOAT)
        return dat[:,:,0].swapaxes(0,1).reshape(dat[:,:,0].shape,order='F').T, R


    def load_textures(self):
        #global textures
        
        textureSurface = pygame.image.load(os.path.join('/Users/yves/Desktop/nehe1-10/data','9pk.jpg'))
        textureData = pygame.image.tostring(textureSurface, "RGBX", 1)  #convert texturedata to string

        textureSurface2 = pygame.image.load(os.path.join('/Users/yves/Desktop/nehe1-10/data','pvc.jpg'))
        textureData2 = pygame.image.tostring(textureSurface2, "RGBX", 1)  #convert texturedata to string

        self.textures = [glGenTextures(1),glGenTextures(1)]     #define one texture
        
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)#
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureSurface.get_width(), textureSurface.get_height(), 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, textureData )
        
        glBindTexture(GL_TEXTURE_2D, self.textures[1])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)#
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureSurface2.get_width(), textureSurface2.get_height(), 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, textureData2 )



    def reset(self):
        self.xpos =0.
        self.ypos = 0.5
        self.zpos = 0.

    def setup_world(self):
        #global tris
        verts = 0
        tri = []
        self.tris = []
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
                self.tris.append(tri)
                tri = []
                verts = 0
        #print(len(tris))

    def resize(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, float(self.width)/self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1, 1, 1, .5]);

        glLoadIdentity()




def init(module, weight_init, bias_init, gain=1):
    """
    :param module: module to initialize
    :param weight_init: initialization scheme
    :param bias_init: bias initialization scheme
    :param gain: gain for weight initialization
    :return: initialized module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Policy(nn.Module):
    def __init__(self,window_size,num_filter=2,size=2,pad=0,stride=2,lstm_hidden=0):
        super(Policy, self).__init__()


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('leaky_relu'))

        self.num_filter = num_filter
        self.stride = stride
        self.size = size
        self.pad =pad

        self.conv1 = nn.Conv2d(1, self.num_filter, self.size,self.stride)
        self.conv2 = nn.Conv2d(self.num_filter, self.num_filter, self.size,self.stride)
        #self.conv3 = nn.Conv2d(self.num_filter, self.num_filter, self.size,self.stride)
        #self.conv4 = init_(nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad))

        #self.affine1 = nn.Linear(np.prod(window_size),28,bias=True)
        #self.dropout = nn.Dropout(p=0.1)


        self.lstm_hidden = lstm_hidden
        if self.lstm_hidden:
            self.lstm = nn.LSTMCell(input_size=882, hidden_size=self.lstm_hidden)
            self.h_1 = self.c_1 = torch.zeros([1,self.lstm_hidden])

            self.affine2 = nn.Linear(self.lstm_hidden, 6,bias=True)
        else:
            self.affine2 = nn.Linear(882, 6,bias=True)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        #x = self.affine1(x)
        #x = self.dropout(x)
        #x = F.relu(x)
        #action_scores = self.affine2(x)
        #return F.softmax(action_scores, dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))

        if self.lstm_hidden:
            self.h_1, self.c_1 = self.lstm(x.view([1,-1]), (self.h_1, self.c_1))
            action_scores = self.affine2(self.h_1)

        else:
            action_scores = self.affine2(x.view([1,-1]))



        return F.softmax(action_scores, dim=1)



    def reset_lstm(self, buf_size=None, reset_indices=None):
        """
        Resets the inner state of the LSTMCell
        """
        if self.lstm_hidden:
            with torch.no_grad():
                #if reset_indices is None:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                self.h_1 = self.c_1 = torch.zeros([1,self.lstm_hidden])
                #else:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                    #resetTensor = torch.as_tensor(reset_indices.astype(np.uint8))

                    #if resetTensor.sum():
                    #    self.h_t1 = (1 - resetTensor.view(-1, 1)).float() * self.h_t1
                    #    self.c_t1 = (1 - resetTensor.view(-1, 1)).float() * self.c_t1


def select_action(state):
    #state = torch.from_numpy(state).float().unsqueeze(0)
    #probs = policy(state)
    probs = policy(torch.Tensor(state).view([1,1,window_size[0],window_size[1]]))
    #print(probs)
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
    policy_loss.backward(retain_graph=True)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]




if __name__=='__main__':
    window_size = (86,86)
    env = poke_env(window_size=window_size,fov=130)

    nFrames_run = 0

    policy = Policy(window_size,lstm_hidden=50)
    policy.reset_lstm()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    eps = np.finfo(np.float32).eps.item()


    st = time.time()

    allR = []
    action = 0
    #for t_ in range(1,3000):
    t_ = 1
    allA_tmp = []
    showA = 0

    reset_every = 201
    avgR = 0
    alpha_= 0.999
    while True:

        dat,rew = env.draw()
        

        action = select_action(dat/255.)
        allA_tmp.append(action)
        env.handle_keys(action)

        #all_actions.append(action)
        #handle_keys(action)
        allR.append(rew)
        avgR = alpha_*avgR + (1-alpha_)*rew
        policy.rewards.append(rew)

        if np.remainder(t_,51)==0:
            #print('update')
            finish_episode()
            nFrames_run = 0
            st= time.time()
            showA = np.array([allA_tmp.count(i) for i in range(6)])/len(allA_tmp)
            allA_tmp = []
            policy.h_1 = policy.h_1.detach()
            policy.c_1 = policy.c_1.detach()


        event = pygame.event.poll()
        if event.type == KEYDOWN:
            if event.key==K_UP:
                env.dist_thresh = np.clip(env.dist_thresh+0.001,.1,10)
            if event.key==K_DOWN:
                env.dist_thresh = np.clip(env.dist_thresh-0.001,.1,10)
            if event.key==K_LEFT:
                env.move_gain = np.clip(env.move_gain-0.0001,.001,.5)
            if event.key==K_RIGHT:
                env.move_gain = np.clip(env.move_gain+0.0001,.001,.5)
            if event.key==K_1:
                reset_every = np.clip(reset_every+10,1,10000000)
            if event.key==K_2:
                reset_every = np.clip(reset_every-10,1,10000000)




        nFrames_run += 1
        t_ += 1
        sys.stdout.write("\rtotF:{}  |fps:{:.3f}   |   muR:{:.6f}   |   muA:{}       |dist_thresh:{:.4f}    |reset_every:{}       |move_gain:{:.4f} ".format(t_,nFrames_run/(time.time()-st),
                                                avgR,
                                                0,
                                                env.dist_thresh,
                                                reset_every,
                                                env.move_gain)
                        )

        if np.remainder(t_,reset_every)==0:
            env.reset()
            policy.reset_lstm()
        sys.stdout.flush()
        pygame.display.flip()
