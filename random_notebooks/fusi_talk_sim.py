import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn import linear_model
import itertools
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D 


if __name__ == '__main__':

    #np.random.seed(80) #set seed to ensure correct rotation of MDS plot
    nNeurons = 200

    #weights to three different mearninful categories
    WR = np.random.normal(size=(nNeurons)) #reward weights

    WC = np.random.normal(size=(nNeurons)) #context weights

    WA = np.random.normal(size=(nNeurons)) #action weights


    #assume all variables are binary; get all possible combinations of variables
    all_comb = []
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                all_comb.append([i,j,k])


    #set number of trials
    nTrials = 5000


    #Get activities across conditions
    activities=  []
    conditions = []
    for trlNr in range(nTrials):
        samp_cond = all_comb[np.random.choice(np.arange(len(all_comb)))]  #select a combination of trial types at random
        conditions.append(samp_cond) #append to list
        
        #sample neural activity given the sampled condition
        activity = np.random.normal(WR*samp_cond[0] + WC*samp_cond[1] + WA*samp_cond[2],scale=1)  
        
        activities.append(activity)    #append activity

        
    activities = np.array(activities)
    conditions = np.array(conditions)



    #Test 
    lg = linear_model.LogisticRegression()


    #select all trials in which the second thing (i.e. context) is 1
    trials01 = np.where(np.logical_and(conditions[:,0]==0,conditions[:,1]==1))[0]
    trials11 = np.where(np.logical_and(conditions[:,0]==1,conditions[:,1]==1))[0]


    #select all trials in which the second thing (i.e. context) is 0
    trials00 = np.where(np.logical_and(conditions[:,0]==0,conditions[:,1]==0))[0]
    trials10 = np.where(np.logical_and(conditions[:,0]==1,conditions[:,1]==0))[0]


    # learn to predict whether reward was delivered in context 1
    fitted = lg.fit(np.concatenate([activities[trials01],activities[trials11]]),
                    np.concatenate([np.ones(len(trials01)),np.zeros(len(trials11))]))

    #test reward predictions in context 0
    generalisation_performance = fitted.score(np.concatenate([activities[trials00],activities[trials10]]),
                                              np.concatenate([np.ones(len(trials00)),np.zeros(len(trials10))]))

    print("cross-condition generalisation performance: {:.3f}%".format(100*generalisation_performance))


    #Do MDS plot

    mds  = MDS(n_components=3)
    out = mds.fit_transform(activities)
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')

    ix = 1

    ax.scatter(out[conditions[:,ix]==1,0], out[conditions[:,ix]==1,1], out[conditions[:,ix]==1,ix], marker='o',label='context 1')
    ax.scatter(out[conditions[:,ix]==0,0], out[conditions[:,ix]==0,1], out[conditions[:,ix]==0,ix], marker='o',label='context 2')

    ax.set_xlabel('MDS Dim 1')
    ax.set_ylabel('MDS Dim 2')
    ax.set_zlabel('MDS Dim 3')
    plt.legend()

    ax.view_init(10, -15)
    plt.show()

