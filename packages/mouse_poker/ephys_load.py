import os
import sys
import numpy as np
import re

def get_binned_spikes(spks,bin_size=10):
    """ Takes in spike times of units and returns binned spike
        rates (no smoothing) at specified resolution. Bin size
        is in ms
    """

    #30 because sampling rate of the ephys is 
    maxT = (np.nanmax([np.nanmax(i) for i in spks])/30.)/bin_size

    spk_arr = np.zeros([len(spks),int(np.ceil(maxT))])
    for i,u in enumerate(spks):
        spk_arr[i,np.floor(u/30/bin_size).astype("int")[:,0]] = 1
    
    return spk_arr

def get_port_to_state_map(dat_dict,task_times):
    task1 = []
    task2 = []
    for dp,ds in zip(dat_dict['port'],dat_dict['state']):
            
        if np.any([np.logical_and((dp[2]*1000)>t_[0],(dp[2]*1000)<t_[1]) for t_ in task_times[0]]):
            tmp = [dp[0],ds[0]]
            if tmp not in task1:
                task1.append(tmp)
        else:
            tmp = [dp[0],ds[0]]
            if tmp not in task2:
                task2.append(tmp)
        
    return task1,task2

def get_task_ranges(lines,task_nr,rsync_times_pyc):
    """ Return task ranges in ms in the reference frame of 
        the behaviour system
    """

    task_switches = [re.findall(' ([0-9]*)', i)[0] for i in lines if 'change_task' in i and i.startswith('V')]
    task_boundU = [int(re.findall(' ([0-9]*) ',lines[ix+1])[0]) for ix,l_ in enumerate(lines) if "OLD" in l_] + [np.max(rsync_times_pyc)]
    task_boundL = [0] + [int(re.findall(' ([0-9]*) ',lines[ix+1])[0]) for ix,l_ in enumerate(lines) if "NEW" in l_]
    task_order = [int(re.findall('([0-9])\n',lines[ix+3])[0]) for ix,l_ in enumerate(lines) if "OLD" in l_]
    if task_order[0]==2:
        task_order = [1] + task_order
    elif task_order[1]==2:
        task_order = [2] + task_order

    cTask = int(task_nr)
    task1_times = []
    task2_times = []
    for tl_,tu_ in zip(task_boundL,task_boundU):

        if cTask==1:
            task1_times.append([tl_,tu_])
        elif cTask==2:
            task2_times.append([tl_,tu_])
        if cTask==1: cTask=2
        elif cTask==2: cTask=1
    task_times = [task1_times,task2_times]
    return task_times


def get_unit_spike_lists(spkT,spkC,unit_ids=None):
    """ Takes in essentially the kilosort/phy output and returns
        lists with the spike times per unit
        
        Arguments:
        ===================================
        
        spkT: spike times
        
        spkC: cluster membership of each spike
        
        unit_ids: units you want to sort
    """
    
    if unit_ids is None:
        unit_ids = np.arange(len(np.unique(spkC)))
    
    spks = []
    for uid in unit_ids:
        tmp = spkT[np.where(spkC==uid)[0]]
        spks.append(tmp[np.where(np.logical_not(np.isnan(tmp)))[0]])
    return spks

def build_task_DM(task_times,maxT,bin_size):
    
    task_states = len(task_times)
    task_DM = np.zeros([task_states,maxT])
    for task_nr,times in enumerate(task_times):
        for t0_,t1_ in times:
            t0_ = int(np.floor(t0_/bin_size)); t1_ = int(np.ceil(t1_/bin_size))
            task_DM[task_nr,t0_:t1_] = 1
    return task_DM

def get_lights_off_dm(events,event_times,maxT,bin_size=10):
    light_off_ixs = np.where(events=='lights_off')[0]
    light_off_ts = event_times[light_off_ixs]
    
    handle_poke_ts = event_times[np.where(events=='handle_poke')[0]]
    lights_off_DM = np.zeros(maxT)
    for lot in light_off_ts:
        tmp = handle_poke_ts - lot
        tmp = tmp[tmp>0]
        next_handle_poke = np.min(tmp) +lot
        
        lights_off_DM[int(np.floor(lot/bin_size)):int(np.floor((next_handle_poke)/bin_size))] = 1
    
    return lights_off_DM

def build_state_DM(port_DM,task_DM,task1_map,task2_map):
    
    task1p = [i[0] for i in task1]
    task1s = [i[1] for i in task1]; task1s = np.array(task1s) - np.min(task1s)
    
    task2p = [i[0] for i in task2]
    task2s = [i[1] for i in task2]; task2s = np.array(task2s) - np.min(task2s)
    
    state_DM_task1 = np.zeros([port_DM.shape[0]+1,port_DM.shape[1]])
    state_DM_task2 = np.zeros([port_DM.shape[0]+1,port_DM.shape[1]])

    for t_,col in enumerate(port_DM.T):
        if np.sum(col):
            tmp2 = [int(i) for i in np.where(col)[0]] #which port poked. Loop because sometime 2 ports poked in small window (i.e. one from paw and one snout)
            for tmp in tmp2:
                if tmp in task1p:
                    state_DM_task1[task1s[task1p.index(tmp)],t_] = 1
                else:
                     state_DM_task1[-1,t_] = 1

                if tmp in task2p:
                    state_DM_task2[task2s[task2p.index(tmp)],t_] = 1
                else:
                     state_DM_task2[-1,t_] = 1
    return state_DM_task1,state_DM_task2   

def get_port_to_state_map(dat_dict,task_times):
    
    task1 = []
    task2 = []
    for dp,ds in zip(dat_dict['port'],dat_dict['state']):
            
        if np.any([np.logical_and((dp[2]*1000)>t_[0],(dp[2]*1000)<t_[1]) for t_ in task_times[0]]):
            tmp = [dp[0],ds[0]]
            if tmp not in task1:
                task1.append(tmp)
        else:
            tmp = [dp[0],ds[0]]
            if tmp not in task2:
                task2.append(tmp)
        
    return task1,task2

def build_port_DM(event_times,events,dat_dict,maxT,bin_size):
    
    bin_mult = 1000/float(bin_size)
    port_DM = np.zeros([9,maxT])
    dd_times = np.array([i[2] for i in dat_dict['port']])
    for et,en in zip(event_times,events):
        if ('poke_' in en) and ('out' not in en):
            bin_time = int(np.floor(et*bin_mult))
            port_DM[int(en[-1])-1,bin_time] = 1

    return port_DM

def build_state_DM(port_DM,task_DM,task1_map,task2_map):
    
    task1p = [i[0] for i in task1_map]
    task1s = [i[1] for i in task1_map]
    
    task2p = [i[0] for i in task2_map]
    task2s = [i[1] for i in task2_map]
    
    state_DM_task1 = np.zeros([port_DM.shape[0]+1,port_DM.shape[1]])
    state_DM_task2 = np.zeros([port_DM.shape[0]+1,port_DM.shape[1]])

    for t_,col in enumerate(port_DM.T):
        if np.sum(col):
            tmp2 = [int(i) for i in np.where(col)[0]] #which port poked. Loop because sometime 2 ports poked in small window (i.e. one from paw and one snout)
            for tmp in tmp2:
                if tmp in task1p:
                    state_DM_task1[task1s[task1p.index(tmp)],t_] = 1
                else:
                     state_DM_task1[-1,t_] = 1

                if tmp in task2p:
                    state_DM_task2[task2s[task2p.index(tmp)],t_] = 1
                else:
                     state_DM_task2[-1,t_] = 1
    return state_DM_task1,state_DM_task2   