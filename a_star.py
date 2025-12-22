from env.MazeEnv import *
from models.AStar import a_star_maze_solve 

env = MazeEnv("../C/c-qlearning/gwyn_1.maze")

print(f"{env.agent_start} --> {env.agent_goal}")
print(f"Walls count: {env.walls_count}")

path = a_star_maze_solve(env)

print("Path found: ", path)
print("Distance to: ",path.distance_to(path))