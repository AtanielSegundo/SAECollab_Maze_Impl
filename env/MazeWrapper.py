import numpy as np

from enum import Enum
from typing import *
from collections import deque

from .MazeEnv import MazeEnv,Action,GridCell

'''
[Ataniel - 02/12/2025]

NA HORA DE TREINAR O MODELO TEMOS DUAS COIASAS A CONSIDERAR:

1 - O ENCODING DO ESTADO NO AMBIENTE
2 - FEATURES EXTRAS AO ESTADO:
    a - N ESTADOS ANTERIORES (ULTIMOS ESTADOS)
    b - N ACOES   ANTERIORES (ULTIMAS ACOES TOMADAS)
    C - ACOES     POSSIVEIS  (SENSORES INDICANDO SE TEM PAREDES OU NÃO')

E NECESSARIO CONCATENAR O ESTADO COM AS FEATURES EXTRAS

a - np.array([last_row,last_col],dtype=np.float32).reshape(1,-1)
b - last_action = np.array(list(Action)[last_action_idx].value).reshape(1,-1)
c - [is_left,is_right,is_up,is_down].reshape(1,-1)

'''

class StateEncoder(Enum):
    COORDS = "coords"
    COORDS_NORM = "coords_norm"
    ONE_HOT = "one_hot"

    def encode(self, row, col, n_rows, n_cols):
        if self is StateEncoder.COORDS:
            return np.array([[row, col]], dtype=np.float32)

        elif self is StateEncoder.COORDS_NORM:
            base = StateEncoder.COORDS.encode(row, col, n_rows, n_cols)
            return base * np.array([[1/(n_rows-1), 1/(n_cols-1)]], dtype=np.float32)

        elif self is StateEncoder.ONE_HOT:
            idx = row * n_cols + col
            return np.eye(n_rows * n_cols, dtype=np.float32)[idx].reshape(1, -1)

        else:
            raise ValueError("invalid encoder")
    
    def decode(self,arr,n_rows, n_cols):
        row,col = 0,0

        if self is StateEncoder.COORDS:
            row, col = int(arr[0, 0]), int(arr[0, 1])
        elif self is StateEncoder.COORDS_NORM:
            row = int(arr[0, 0] * (n_rows - 1))
            col = int(arr[0, 1] * (n_cols - 1))
        elif self is StateEncoder.ONE_HOT:
            idx = np.argmax(arr[0])
            row = idx // n_cols
            col = idx % n_cols
        else:
            raise ValueError("Unknown state encoder")
        
        return row,col
        
class MazeGymWrapper:
    def __init__(self, maze: MazeEnv, 
                 state_encoder=StateEncoder.COORDS,
                 num_last_states:int=None,  #Number of before state
                 num_last_actions:int=None, #Number of before actions
                 possible_actions_feature:bool=False, #Enable check neighbours features
                 visited_count:bool = False,
                 ):
        self.maze = maze

        r_start = int(self.maze.agent_start[0])
        c_start = int(self.maze.agent_start[1])
        self.start = (r_start, c_start)

        r_goal = int(self.maze.agent_goal[0])
        c_goal = int(self.maze.agent_goal[1])
        self.goal = (r_goal, c_goal)

        self.state = self.start
        self.state_size = 0
        self.state_encoder = state_encoder
        self.rows = self.maze.rows
        self.cols = self.maze.cols
        self.action_size = len(list(Action))

        self.last_states  = deque(maxlen=num_last_states) if num_last_states else None
        self.last_actions = deque(maxlen=num_last_actions) if num_last_actions else None
        self.possible_actions_feature = possible_actions_feature
        self.visit_count  = np.zeros(shape=(self.rows,self.cols),dtype=np.float32) if visited_count else None

        if self.state_encoder is StateEncoder.ONE_HOT:
            self.state_size = self.rows * self.cols
        else:
            self.state_size = 2
            if self.last_states is not None:
                self.state_size += 2 * self.last_states.maxlen
        
        if self.last_actions is not None:
            self.state_size += 2 * self.last_actions.maxlen
        if self.possible_actions_feature:
            self.state_size += len(list(Action))
        
        # if self.visit_count is not None:
        #     self.state_size += 1

    def encode(self, rc: Tuple[int,int]) -> np.ndarray:
        """
        Retorna sempre um np.ndarray 1-D shape (state_size,), dtype float32.
        """
        r, c = rc
        enc = self.state_encoder.encode(r, c, self.rows, self.cols)
        enc = np.asarray(enc, dtype=np.float32)
        if enc.ndim == 2 and enc.shape[0] == 1:
            enc = enc.reshape(-1)
        elif enc.ndim == 1:
            pass
        else:
            enc = enc.reshape(-1)
        return np.copy(enc)

    def decode(self,arr:np.ndarray) -> Tuple[int,int]:
        return self.state_encoder.decode(arr,self.rows,self.cols)

    def out_state_features(self) -> np.ndarray:
        state_encoded = self.encode(self.state).reshape(1, -1)
        #print(f"Base state shape: {state_encoded.shape}")

        if self.last_states is not None:
            #print(f"last_states length: {len(self.last_states)}, maxlen: {self.last_states.maxlen}")
            for i in range(self.last_states.maxlen):
                if i < len(self.last_states):  
                    if self.state_encoder is StateEncoder.ONE_HOT:
                        last_encoded = self.encode(self.last_states[i]).reshape(1, -1)*(0.5**i)
                        state_encoded += last_encoded 
                    else:
                        last_encoded = self.encode(self.last_states[i]).reshape(1, -1)
                        #print(f"Adding last_state {i}: {last_encoded}")
                        state_encoded = np.concatenate([state_encoded, last_encoded], axis=1)
                else:
                    if self.state_encoder is StateEncoder.ONE_HOT:
                        state_encoded += np.zeros((1, self.rows * self.cols), dtype=np.float32)
                    else:
                        #print(f"Padding last_state {i} with zeros")
                        state_encoded = np.concatenate([state_encoded, np.zeros((1, 2), dtype=np.float32)], axis=1)

        if self.last_actions is not None:
            #print(f"last_actions length: {len(self.last_actions)}, maxlen: {self.last_actions.maxlen}")
            for i in range(self.last_actions.maxlen):
                if i < len(self.last_actions):  
                    action_delta_reshaped = np.array(self.last_actions[i], dtype=np.float32).reshape(1, -1)
                    #print(f"Adding last_action {i}: {action_delta_reshaped}")
                    state_encoded = np.concatenate([state_encoded, action_delta_reshaped], axis=1)
                else:
                    #print(f"Padding last_action {i} with zeros")
                    state_encoded = np.concatenate([state_encoded, np.zeros((1, 2), dtype=np.float32)], axis=1)

        if self.possible_actions_feature:
            action_arr = []
            for action in list(Action):
                delta = action.delta
                potential_next = (self.state[0] + delta[1], self.state[1] + delta[0])
                
                cell_t = self.maze.getStateGridCell(potential_next)
                
                if cell_t is GridCell.WALL:
                    action_arr.append(0.0)
                else:
                    action_arr.append(1.0)

            actions_reshaped = np.array(action_arr, dtype=np.float32).reshape(1, -1)
            state_encoded = np.concatenate([state_encoded, actions_reshaped], axis=1)

        # if self.visit_count is not None:
        #     visits_count = self.visit_count[*self.state].reshape(1,-1)
        #     state_encoded = np.concatenate([state_encoded, visits_count], axis=1)

        #print(f"Final state shape: {state_encoded.shape}")
        result = state_encoded.reshape(-1)
        #print(f"Returned state: {result}")

        return result

    def reset(self) -> np.ndarray:
        self.state = self.start
        if self.last_actions:
            self.last_actions.clear()
        if self.last_states:
            self.last_states.clear()
        if self.visit_count is not None:
            self.visit_count = np.zeros(shape=(self.rows,self.cols),dtype=np.float32)
        return self.out_state_features()

    def step(self, action_idx: int):
        """
        action_idx -> int
        Retorna: (next_state_vec, reward:float, done:bool, info:dict)
        next_state_vec: np.ndarray 1-D shape (state_size,), dtype float32
        info['raw_ns'] -> (r,c) tuple (inteiros)
        """
        action = list(Action)[action_idx]
        res = self.maze.step(self.state, action)
        next_state_coords = tuple(res.nextState)
        
        if self.visit_count is not None:
            if next_state_coords[0] != self.state[0] and  next_state_coords[1] != self.state[1]:
                self.visit_count[*next_state_coords] += 1 

        if self.last_states is not None:
            self.last_states.append(next_state_coords)
        if self.last_actions is not None:
            self.last_actions.append(action.delta)
        
        reward = float(res.reward)
        if self.visit_count is not None:
            reward *= (1.0 + self.visit_count[*next_state_coords])
        done = bool(res.isGoal)

        self.state = next_state_coords

        out_state_features = self.out_state_features()
        return out_state_features, reward, done, {"raw_ns": next_state_coords}

    def getState(self,row:int,col:int):
        return self.maze.grid[row,col]

    def isGoal(self,rc:tuple[int,int]):
        if rc[0] == self.goal[0] and rc[1] == self.goal[1]:
            return True
        return False