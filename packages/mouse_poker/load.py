import re
import os
import sys
import numpy as np
import pandas as pd
from itertools import chain
import datetime

get_mouse_id = lambda x: re.findall('_([0-9]{6}_.*?)_-',x)
get_date = lambda x: re.findall('_-*(202[0-9]-[0-9]{2}-[0-9]{2})',x)





def create_base_df(ROOT,verbose=True,rerun=False,min_sess_dur=600,dirpath=None):

    """ Creates a pandas dataframe that summarises information about the session that a
        given mouse performed 
        
        TODO: run through everything with os walk
    """
    try:
        df_old = load_df(ROOT)
    except Exception:
        pass 
    df = pd.DataFrame(columns=['mouse_ID','group','date','time','task_name','structure','layout','valid','nRews','task_nr','test','filepath'])
    for folder in os.listdir(ROOT):

        if 'test' not in folder and os.path.isdir(os.path.join(ROOT,folder)):


            #This is a monkey patch to deal with different folder structures
            if dirpath is None:
                dirpath = os.path.join(ROOT,folder)
            else:
                dirpath = ROOT
            if os.path.isdir(dirpath):
                for f_ in os.listdir(dirpath):
                    if verbose: print(f_)
                    fpath = os.path.join(dirpath,f_)

                    if os.path.isfile(fpath) and 'json' not in fpath and '.txt' in fpath:

                        if True:#((fpath not in df_old['filepath'].values) or rerun):
                            try:
                                f = open(fpath,'r')
                                lines = f.readlines()
                                out = get_metadata(lines)

                                experiment_name, task_name, subject_id, task_nr, graph,lineloop,date,test,summary_dict = out
                                dat_dict,events,event_times,nRews,_ = parse_data(lines,experiment_name)
                                date = (np.datetime64(date.replace(' ','T').replace('/','-'))).astype('object')



                                if (len(lines)>200 and test==False and np.max(event_times)>min_sess_dur):
                                    valid = True
                                else:
                                    valid = False

                                if ((str(date.date())=='2020-02-11') and (date.time()<datetime.time(14,0,0))):
                                    valid = False
                                dct1 = {'mouse_ID': subject_id if type(subject_id)==str else eval(subject_id),
                                       'date': date.date(),
                                       'time': date.time(),
                                       'structure': lineloop,
                                       'layout': graph,
                                       'valid': valid,
                                       'nRews': nRews,
                                       'task_nr': task_nr,
                                       'task_name':task_name,
                                       'test':test,
                                       'group':os.path.split(folder)[-1],
                                       'filepath':fpath}
                                #dct2 = [{k:v} for k,v in summary_dict.items() if k not in dct1.keys()]

                                dct1["summary_dict"] = summary_dict
                                #dct_f = dict(chain.from_iterable(d.items() for d in (dct1,dct2)))

                                df = df.append(dct1,ignore_index=True)
                            except Exception as e:
                                if verbose:
                                    print('Warning, %s failed to load' %fpath)
                                    print('Error message: %s' %e)
                        else:
                            df = df.append(df_old.loc[df_old['filepath']==fpath],ignore_index=True)

        if ROOT[-1]=='/': ROOT = ROOT[:-1]

        df.to_json(os.path.join(os.path.split(os.path.realpath(__file__))[0],'df_overview' + os.path.split(ROOT)[1] + '.json'))

    return df

def load_df(ROOT):

    if ROOT[-1]=='/': ROOT = ROOT[:-1]

    fp = os.path.join(os.path.split(os.path.realpath(__file__))[0],'df_overview' + os.path.split(ROOT)[1] + '.json')
    if os.path.isfile(fp):
        df = pd.read_json(fp)
    else:
        raise Exception("Run load.create_base_df(ROOT) first sot hat you have something to load")
    return df




def get_metadata(lines):
    """ Get metadata from the beginning of the file """
    
    summary_lines = []
    experiment_name = task_name = subject_id = task_nr = graph = lineloop = date = test = None
    for l in lines:
        if re.findall('I Experiment name  : (.*?)\n',l): experiment_name = re.findall('I Experiment name  : (.*?)\n',l)[0]
        
        if re.findall('Task name : (.*?)\n',l): task_name = re.findall('Task name : (.*?)\n',l)[0]
        
        if re.findall('Subject ID : (.*?)\n',l): subject_id = eval(re.findall('Subject ID : (.*?)\n',l)[0])
            
        if re.findall('Start date : (.*?)\n',l): date = re.findall('Start date : (.*?)\n',l)[0]
        
        #################################################################################
        #### Specific to this task
        #################################################################################

        if re.findall('V.*? task_nr (.*?)\n',l): task_nr = re.findall('V.*? task_nr (.*?)\n',l)[0]
            
        if re.findall('P.*? (G[0-9]_[0-9])\n',l): graph = re.findall('P.*? (G[0-9]_[0-9])\n',l)[0]

        if re.findall('P .* (LOOP|LINE|loop|line)\n',l): lineloop = re.findall('P .* (LOOP|LINE|loop|line)\n',l)[0].lower()
        if re.findall('TEST ([A-z]*)\\n',l): test = eval(re.findall('TEST ([A-z]*)\\n',l)[0])

        if re.findall('V -1 *',l):
            summary_lines.append(l)

    summary_dict = _get_summary_dict(summary_lines)
    return experiment_name, task_name, subject_id, task_nr, graph, lineloop, date, test, summary_dict


def _get_summary_dict(summary):
    summary_dict = {}
    for i in summary:
        try:

            k =re.findall('-1 (.*?) ',i)[0]
            if '\n' in i:
                summary_dict[k] = eval(re.findall('-1 .*? (.*?)\n',i)[0]) if (k!='subject_id' and k!='graph_type') else re.findall('-1 .*? (.*?)\n',i)[0]
            else:
                summary_dict[k] = eval(re.findall('-1 .*? (.*$)',i)[0]) if (k!='subject_id' and k!='graph_type') else re.findall('-1 .*? (.*$)',i)[0]
        except Exception as e:
            print('exception in _get_summary_dict')
            print(e)
            print(i)
    return summary_dict

def _parse_dat(text):
    """ function that takes data in and returns meaningful stuff """

    if 'POKEDPORT' in text:
        now = int(re.findall('POKEDPORT_([0-9]{1,2})',text)[0])
        avail = eval(re.findall('_NOWPOKES_(\[.*?\])_',text)[0])[0]
        prev = eval(re.findall('_PREVPOKE[S]?_([\[]?.*?[\]]?)_',text)[0])

        dtype = 'port'
    elif 'NOWSTATE' in text:
        now = int(re.findall('_POKEDSTATE_([0-9]{1,2})',text)[0])  #YW change for NAVI used to be '_NOWSTATE_([0-9]{1,2})'
        avail = eval(re.findall('_AVAILSTATES_(\[.*?\])',text)[0])
        prev = eval(re.findall('_PREVSTATE[S]?_([\[]?.*?[\]]?)_',text)[0])
        dtype = 'state'
    else:
        print("WARNING the following line was not processed")
        print(text)

    if 'PROBE' in text:
        probe = eval(re.findall('PROBE_([A-z]*?)_',text)[0])
    else:
        probe = None

    if type(prev)==list:
        if now in prev: 
            teleport=False
        else:
            teleport = True
    else:
        teleport = False

    return now,avail,dtype,teleport,probe,prev


def parse_data(lines,experiment_name):
    start_read = False #placeholder variable that ignores the period where just free rewards are available
    event_times = []
    events = []
    alldat = []
    dat_times = []
    dat_dict = {'state': [],
                'port': [],
                'random': [],
                'rew_locations': [],
                'rews': [],
                'rew_list': [] }


    tot_pokes = 0
    nRews = 0

    text2 = ''.join(lines)
    state_dict0 = eval(re.findall('({.*)\n',text2)[0])
    state_dict = dict([(v,k) for k,v in state_dict0.items()])
    event_dict = eval(re.findall('({.*)\n',text2)[1])
    event_dict = dict([(v,k) for k,v in event_dict.items()])


    ##NEW CODE##
    event_dict = {**state_dict,**event_dict}
    ##END NEW CODE ##
    #print(event_dict.keys())
    rew_list = []
    for ln,l in enumerate(lines):
        try:

            if (str(state_dict0['handle_poke'])+'\n' in l and  l[0]=='D'):
                start_read = True

            if l[0]=='D':
                if start_read:
                    tmp = float(re.findall('D ([-0-9]*)',l)[0])/1000.
                    ev = int(re.findall('D [-0-9]* ([0-9]{1,3})',l)[0])
                    #print(ev)
                    #print(event_dict[ev])
                    if ev in list(event_dict.keys()):
                        #print(event_dict[ev],ev)
                        events.append(event_dict[ev])
                        event_times.append(tmp)

                        if event_dict[ev][-1] in [str(i) for i in range(9)]:
                            tot_pokes += 1
                    #event_dict.keys()

            elif l[0]=='P':
                if 'POKEDSTATE' in l:
                    start_read = True
                

                if start_read:
                    tmp_t = float(re.findall('P ([0-9]*)',l)[0])/1000.
                    dat = re.findall('P [-0-9]* (.*)\n',l)[0]
                    if 'POKE' in dat and 'TELEPORT' not in dat:

                        now,avail,dtype,teleport,probe, prev = _parse_dat(dat)

                        if 'NAVI' in experiment_name:  #NEW YW 11 MARCH 21
                            if dtype=='state':
                                if dat_dict['rew_list'][-1]:
                                    dat_dict['state'].append([prev,avail,tmp_t,None])

                        if dtype=='port':
                            if '_REW_True' in l:
                                dat_dict['rew_list'].append(1)
                            else:
                                dat_dict['rew_list'].append(0)


                        #tmp = re.findall("RANDOM_([A-z]*?)_",dat)
                        #if tmp: dat_dict['random'].append(eval(tmp[0]))
                        if 'POKEDSTATE' in dat:
                            dat_dict['random'].append(teleport)

                        #if teleport==False:
                        dat_dict[dtype].append([now,avail,tmp_t,probe])
                    elif 'REWARD LOCATIONS' in l:
                        #print(l)
                        tmp_t = float(re.findall('P ([0-9]*)',l)[0])/1000.
                        if 'NAVI' in experiment_name:
                            tmp = eval(re.findall('LOCATIONS(.*)',l)[0])
                        else:
                            tmp = eval(re.findall('LOCATIONS(\[.*\])',l)[0])
                        dat_dict['rew_locations'].append([tmp,tmp_t])
                    if '_REW_True' in l:
                        nRews += 1
                        dat_dict['rews'].append([now,tmp_t,probe])
                        #rew_list.append()


        except Exception as e:
            print(l)
            print(ln)
            print(e)
        #    raise Exception
    #print(event_dict.keys())
    return dat_dict, np.array(events), np.array(event_times), nRews, event_dict