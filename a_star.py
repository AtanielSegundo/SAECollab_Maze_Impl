from env.MazeEnv import *
from models.AStar import a_star_maze_solve 

env = MazeEnv("malo.maze")

path = a_star_maze_solve(env)

print("Path found: ", path)
print("Distance to: ",path.distance_to(path))