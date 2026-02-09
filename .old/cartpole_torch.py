import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch

from models.QModels import TorchDDQN


env = gym.make("CartPole-v1")

state_size  = env.observation_space.shape[0]   # 4
action_size = env.action_space.n               # 2

policy = nn.Sequential(
    nn.Linear(state_size, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, action_size)
)

agent = TorchDDQN(
    sequential_list=policy,
    state_size=state_size,
    action_size=action_size,
    lr=5e-4,               
    gamma=0.99,
    batch_size=512,        
    buffer_size=50_000,
    epsilon_start=1.0,
    epsilon_final=0.01,
    epsilon_decay=3000,
    target_update=1000,
    learn_interval=1,
    tau=0.005,
    grad_clip=10.0,
    min_replay_size=1000
)

EPISODES = 1000
MAX_STEPS = 500

def normalize(s):
    return s / np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32)

for ep in range(1, EPISODES+1):
    s, _ = env.reset()
    s = s.astype(np.float32).reshape(-1)
    total = 0

    for t in range(MAX_STEPS):
        a = agent.act(s, eval=False)

        ns, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        ns = ns.astype(np.float32).reshape(-1)
        agent.remember(s, a, r, ns, done)

        s = ns
        total += r
        if done:
            break

    agent.update_epsilon()

    if ep % 10 == 0:
        print(f"Ep {ep}  reward={total:.1f}  eps={agent.epsilon:.3f}")

s, _ = env.reset()
s = s.astype(np.float32).reshape(1,-1)
done = False
score = 0

while not done:
    a = agent.act(s, eval=True)
    s, r, terminated, truncated, _ = env.step(a)
    s = s.astype(np.float32).reshape(1,-1)
    done = terminated or truncated
    score += r

print("Score final:", score)