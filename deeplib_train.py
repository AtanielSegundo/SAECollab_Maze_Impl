import numpy as np
import sys

from MazeEnv import *
from QModels import AStarQModel
from DeepLib.DeepLib import DDQN,layers

file_path = sys.argv[-1]
if not os.path.exists(file_path):
    print(f"[ERROR] File Not Found: {file_path}")
    exit(-1)
base_name = os.path.basename(file_path).split(".")[0]

raw_env = MazeEnv(file_path)

train_state_encoder = StateEncoder.ONE_HOT

a_star_path = a_star_maze_solve(raw_env)
env = MazeGymWrapper(raw_env,train_state_encoder)

state_size  = env.state_size
action_size = env.action_size

agent = DDQN()
a_star_model = AStarQModel(raw_env, state_encoder=train_state_encoder)

agent.set_parameter(
    state_size = state_size,
    action_size = action_size,
    memory_size = env.rows*env.cols*100,
    alpha = 0.07,
    gamma = 0.99,
    policy = agent.exp_epsilon_greedy,
    epsilon = 1.0,
    epsilon_decay = 0.999,
    epsilon_min = 0.01,
    batch_size = 32,
    loss_function = agent.Huber_Loss
)

agent.add_layer(
    layers.Relu(state_size, 32, "xavier"),
    layers.Relu(32, 32, "xavier"),
    layers.Linear(32, action_size, "xavier")
)

EPISODES  = 2000
MAX_STEPS = env.rows * env.cols * action_size

for ep in range(EPISODES):
    state = env.reset()   # shaped (1,2)
    for t in range(MAX_STEPS):
        action = agent.act(state)              # retorna int
        next_state, reward, done, info = env.step(action)
        agent.memorizar(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"ep {ep}/{EPISODES} steps {t} eps {agent.epsilon:.3f}", end="\r")
            break
    agent.policy_update()   # reduz/exponential decay do epsilon

path:Path = get_best_path(env, agent, max_steps=env.rows * env.cols * 10)

print("Saving model Q table")

print(action_size)

genearate_qtable_from_model(raw_env,agent,action_size,
                            state_encoder=train_state_encoder,
                            get_q_val_method='__call__').save(f"{base_name}_deeplib.qtable")

genearate_qtable_from_model(raw_env,a_star_model,action_size,
                            state_encoder=train_state_encoder,
                            get_q_val_method='__call__'
                            ).save("malo_a_star.qtable")

print(get_best_path(env,a_star_model,max_steps=env.rows * env.cols * 10))

print("PATH (coords):")
print(path)

print("A STAR PATH:")
print(a_star_path)

print("DISTANCE BETWEN PATHS:")
print(path.distance_to(a_star_path))

print("SIMILARITY BETWEN PATHS:")
print(path.similarity_to(a_star_path))