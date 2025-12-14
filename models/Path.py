import numpy as np

from typing import *
from env.MazeWrapper import MazeGymWrapper

class Path:
     def __init__(self,arr):
        self.arr = np.array(arr) if arr else None

     @property
     def len(self):
        return len(self.arr)
    
     def distance_to(self, other: 'Path') -> float:
        """
        Compute sum of minimum distances from each point in self to any point in other.
        Optimized using vectorized NumPy operations.
        """
        if self.arr is None or other.arr is None: 
            print("[ERROR] TRYING TO USE NONE DATA ARRAY") 
            return float('inf')

        self_points = self.arr[:, np.newaxis, :]    # Shape: (n, 1, d)
        other_points = other.arr[np.newaxis, :, :]  # Shape: (1, m, d)
        
        differences = self_points - other_points    # Shape: (n, m, d)
        distances = np.linalg.norm(differences, axis=2)  # Shape: (n, m)
        min_distances = np.min(distances, axis=1)  # Shape: (n,)
        
        return np.sum(min_distances)
     
     def similarity_to(self, other: 'Path') -> float:
        """
        Computes a similarity metric between 0 and 1.
        
        Returns:
        - 1.0: Paths have same length and pass through same points
        - 0.5-1.0: Paths are similar in size and point distribution
        - < 0.5: Paths differ significantly in length or point distribution
        """
        if self.arr is None or other.arr is None:
            print("[ERROR] TRYING TO USE NONE DATA ARRAY")
            return 0.0
        
        len_self = self.len
        len_other = other.len
        
        if len_self == 0 or len_other == 0:
            size_sim = 1.0 if len_self == len_other else 0.0
        else:
            size_sim = min(len_self, len_other) / max(len_self, len_other)
        
        point_distance = (self.distance_to(other) + other.distance_to(self)) / 2
        avg_path_length = (len_self + len_other) / 2 if (len_self + len_other) > 0 else 1.0
        
        # Normalize by average path length to make it scale-invariant
        normalized_distance = point_distance / avg_path_length if avg_path_length > 0 else 0.0
        point_sim = 1.0 / (1.0 + normalized_distance)
        
        # Combine both factors: geometric mean for balance
        similarity = np.sqrt(size_sim * point_sim)
        
        return float(similarity)
     
     def __str__(self,m:MazeGymWrapper=None):
        _str = ["\n"]
        for i in range(self.len):
            if m is None:
                _str.append(f"{i}: {self.arr[i]}\n")
            else :
                _str.append(f"{i}: {self.arr[i]} [{m.maze.getStateGridCell(self.arr[i])}]\n")
        return "".join(_str)
     
def get_best_path(env: MazeGymWrapper, agent, max_steps: int = 10000) -> List[Tuple[int,int]]:
    """
    Retorna lista de tuplas (r,c) seguindo política greedy (argmax de Q).
    Evita loops: se um estado for visitado 4 vezes, aborta.
    Usa forward direto: q = agent(state)
    """
    path = []
    visit_counts = {}
    state_enc = env.reset()
    path.append(env.state)
    visit_counts[env.state] = 1

    for _ in range(max_steps):
        q = agent(state_enc)                 
        q = np.asarray(q)
        if q.ndim == 2:
            q_vals = q[0]
        else:
            q_vals = q.flatten()
        action_idx = int(np.argmax(q_vals))

        state_enc, reward, done, _ = env.step(action_idx)
        path.append(env.state)

        visit_counts[env.state] = visit_counts.get(env.state, 0) + 1
        if visit_counts[env.state] > 4:
            break
        if done:
            break

    return Path(path)