import numpy as np
import heapq

from env.MazeEnv import Action,GridCell,MazeEnv
from env.MazeWrapper import MazeGymWrapper
from models.Path import Path
from collections import deque

INF = 10**9

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

neighbours_deltas = [ Action.LEFT.delta, Action.RIGHT.delta,
                      Action.UP.delta, Action.DOWN.delta
                    ]   

class AStarNode:
   def __init__(self, position, parent=None):
       self.position = position
       self.parent = parent
       self.g = 0
       self.h = 0
       self.f = 0
   def __eq__(self, other):
       return self.position == other.position
   
   def __lt__(self, other):
    return self.f < other.f

def a_star_maze_solve(maze: MazeEnv, 
                      heuristic=manhattan_distance, 
                      neighbours_delta = neighbours_deltas,
                      alt_start = None,
                      alt_end = None,
                       ):
   open_list = []
   closed_list = set()
   start_node = AStarNode(alt_start) if alt_start else AStarNode(maze.agent_start)
   end_node = AStarNode(alt_end) if alt_end else AStarNode(maze.agent_goal)
   heapq.heappush(open_list, (start_node.f, start_node))
   while open_list:
       _, current_node = heapq.heappop(open_list)
       closed_list.add(current_node.position)
       
       if current_node == end_node:
           path = []
           while current_node:
               path.append(current_node.position)
               current_node = current_node.parent
           return Path(path[::-1])
       
       for offset in neighbours_delta:
           neighbor_pos = (current_node.position[0] + offset[0], current_node.position[1] + offset[1])
       
           if not (0 <= neighbor_pos[0] < len(maze.grid) and 0 <= neighbor_pos[1] < len(maze.grid[0])):
               continue # Skip out-of-bounds positions
       
           if maze.grid[neighbor_pos[0]][neighbor_pos[1]] == GridCell.WALL:
               continue # Skip blocked cells
       
           neighbor_node = AStarNode(neighbor_pos, current_node)
           if neighbor_pos in closed_list:
               continue
       
           neighbor_node.g = current_node.g + 1
           neighbor_node.h = heuristic(neighbor_pos, maze.agent_goal)
           neighbor_node.f = neighbor_node.g + neighbor_node.h
           if any(neighbor.position == neighbor_pos and neighbor.f <= neighbor_node.f for _, neighbor in open_list):
               continue
       
           heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
   
   return Path(None)

class AStarQModel:
    """
    Pseudo-model that generates Q-values from A* pathfinding.
    Computes optimal paths from each state to goal using A*, then derives Q-values.
    """
    def __init__(self, maze: MazeGymWrapper):
        self.maze = maze
        self.action_size = 4
        self.build_qcache_by_bfs()
    
    def build_qcache_by_bfs(self):
        maze = self.maze
        n_actions = self.action_size

        rows, cols = maze.rows, maze.cols
        dist = np.full((rows, cols), INF, dtype=np.int32)
        prev_action = np.full((rows, cols), -1, dtype=np.int8)  # action idx that moves toward goal
        self.q_cache = {}

        goal = maze.goal

        dq = deque()
        dist[goal] = 0
        dq.append(goal)

        actions = list(Action)
        while dq:
            r, c = dq.popleft()
            for ai, a in enumerate(actions):
                pr = r - a.delta[1]
                pc = c - a.delta[0]
                if not (0 <= pr < rows and 0 <= pc < cols): 
                    continue
                if maze.getState(pr, pc) == GridCell.WALL.value: 
                    continue
                if dist[pr, pc] != INF: 
                    continue
                dist[pr, pc] = dist[r, c] + 1
                prev_action[pr, pc] = ai
                dq.append((pr, pc))

        for r in range(rows):
            for c in range(cols):
                if maze.getState(r,c) == GridCell.WALL.value: 
                    continue
                q = np.full(n_actions, -100.0, dtype=np.float32)
                best = prev_action[r,c]
                for ai, a in enumerate(actions):
                    nr = r + a.delta[1]; nc = c + a.delta[0]  # match MazeEnv.step convention
                    if 0 <= nr < rows and 0 <= nc < cols and maze.getState(nr,nc) != GridCell.WALL.value:
                        q[ai] = 0.0
                if best >= 0:
                    q[best] = 1.0
                self.q_cache[(r,c)] = q
    
    def forward(self, state_encoded):
        
        if state_encoded.ndim == 1:
            state_encoded = state_encoded.reshape(1, -1)
        
        row, col = self.maze.decode(state_encoded)
        state = (row, col)
        
        if state in self.q_cache:
            return self.q_cache[state].reshape(1, -1)
        else:
            return np.zeros((1, self.action_size), dtype=np.float32)
    
    def __call__(self, state_encoded):
        """Make the model callable, like the neural network agent."""
        return self.forward(state_encoded)