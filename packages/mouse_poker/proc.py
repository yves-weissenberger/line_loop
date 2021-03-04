import numpy as np
import re


def get_pokes(events,event_times,bin_mult=1000):

    """takes in events and return various poke stuff. Importantly, 
       ensures that the number of inpokes and outpokes are matched by, if unbalanced, 
       removing either the last inpoke or the first outpoke 
    """
    inPokes = np.array([int(i[-1])-1 for kk,i in enumerate(events) if re.findall('poke_[0-9]$',i)])
    inPoke_t = np.floor(event_times[np.array([kk for kk,i in enumerate(events) if re.findall('poke_[0-9]$',i)])]*bin_mult).astype('int')    
    outPoke_t = np.floor(event_times[np.array([kk for kk,i in enumerate(events) if re.findall('poke_[0-9]_out',i)])]*bin_mult).astype('int')
    if len(inPoke_t)!=len(outPoke_t):    
        if np.argmin([inPoke_t[0],outPoke_t[0]])==1: outPoke_t = outPoke_t[1:]
        if len(outPoke_t)>len(inPoke_t): outPoke_t = outPoke_t[1:] #if the first recorded thing is an outpoke
        if len(inPoke_t)>len(outPoke_t): inPoke_t = inPoke_t[:-1]; inPokes = inPokes[:-1]
    poke_dur = outPoke_t - inPoke_t

    return inPokes,inPoke_t, outPoke_t,poke_dur
