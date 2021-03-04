import pandas as pd



class experiment(object):
	""" Class aggregating over subjects """



class subject(object):
    """ This class aggregates data over all sessions from a 
    	single subject"""
    
    def __init__(self):

    	self.group = None
        pass



class session(object):
    
    """ This class summarises data for a single behavioral session """
    def __init__(self,fpath):
        self._file_loc = fpath

        self.nRews = None
        self.layout = None
        self.line_loop = None
        
        
    def _load_data(self):
        pass

    def save(self,save_path):
        pass
    
    