# IMPLEMENTATION OF EARLY STOP TRAINING FOLLOWS
import numpy as np 

class early_stop(object):
    def __init__(self, patience=10):
        self.count = 0
        self.patience = patience
        self.best = np.inf #-2 

    def track(self, current):

        if current < self.best:
            self.best = current
            self.count = 0
            save_flag = True
            stop_flag = False
            print('Improved')
        else:
            self.count += 1
            save_flag = False
            print('not Improved for {} - Best was {:.4f}'.format(self.count, self.best))
            if self.count == self.patience:
                stop_flag = True
            else:
                stop_flag = False

        return save_flag, stop_flag
