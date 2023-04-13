import numpy as np

class tracker(object):
    '''
    Add description here
    
    '''
    
    def __init__(self):
        
        self.stored_loss =[]
        self.stored_risk =[]
        self.cum_loss =0.0
        self.iter = 0
        
    def loss_increment(self,loss):
        
        self.stored_loss.append(loss)
        
        self.cum_loss += loss
        self.iter+=1
    
    def risk_increment(self,risk):
        
        self.stored_risk.extend(risk)
        
    def average(self,n_batches):
        print("Tracker iter {} and Data Loader n_batches {}".format(self.iter,n_batches))
        return self.cum_loss/self.iter
        
