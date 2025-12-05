import numpy as np
import struct

from itertools import product as cartesian
from typing import *

from env.MazeWrapper import MazeGymWrapper

class Qtable:
    def __init__(self,n_rows,n_cols,n_actions):
        self.n_rows    = n_rows
        self.n_cols    = n_cols
        self.n_actions = n_actions
        self.q_vals    = np.zeros((n_rows*n_cols*n_actions,),dtype=np.float32)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            f.write(struct.pack('Q', self.n_cols))
            f.write(struct.pack('Q', self.n_rows))
            f.write(struct.pack('Q', self.n_actions))
            self.q_vals.tofile(f)
        
# 'model' should have an method that outputs q-values for actions given an state
def genearate_qtable_from_model(maze:MazeGymWrapper,model,n_actions,get_q_val_method='forward'
                                , use_extras = True
                                ) -> Qtable:
    q_table = Qtable(maze.rows,maze.cols,n_actions)
    for (row,col) in cartesian(range(maze.rows),range(maze.cols)):
        if use_extras:
            original_state = maze.state
            maze.state = (row, col)
            model_in = maze.out_state_features()
            maze.state = original_state
        else :
            model_in = maze.encode((row,col))
        q_vals = getattr(model,get_q_val_method)(model_in)
        q_vals = np.array(q_vals).reshape(-1)
        for i,q_val in enumerate(q_vals):
            q_val_idx = n_actions*(row*maze.cols + col) + i
            q_table.q_vals[q_val_idx] = q_val
    return q_table