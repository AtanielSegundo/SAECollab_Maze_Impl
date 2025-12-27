# QModels.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random as rand
import copy

from StackedCollab.collabNet import SAECollabNet
from collections import deque,namedtuple
from typing import *


Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity:int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size:int):
        batch = rand.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        return batch
    
    def __len__(self):
        return len(self.buffer)
    

def build_network_from_sequential_list(seq_list: Union[nn.Sequential, List, Iterable],
                                       input_dim:int, output_dim:int) -> nn.Module:
    """
    Aceita:
     - nn.Sequential -> retorna (assegura que o último tamanho seja output_dim se necessário)
     - lista de módulos -> nn.Sequential(*list)
     - lista de ints -> cria MLP [input_dim -> hidden1 -> ... -> output_dim]
    """
    if isinstance(seq_list, nn.Sequential):
        net = seq_list
        try:
            dummy = torch.zeros(1, input_dim)
            out = net(dummy)
            if out.shape[-1] != output_dim:
                net = nn.Sequential(net, nn.Linear(out.shape[-1], output_dim))
        except Exception:
            net = nn.Sequential(net, nn.Linear(input_dim, output_dim))
        return net

    if isinstance(seq_list, (list, tuple)) and len(seq_list)>0 and all(isinstance(m, nn.Module) for m in seq_list):
        net = nn.Sequential(*seq_list)
        try:
            dummy = torch.zeros(1, input_dim)
            out = net(dummy)
            if out.shape[-1] != output_dim:
                net = nn.Sequential(net, nn.Linear(out.shape[-1], output_dim))
        except Exception:
            net = nn.Sequential(*seq_list, nn.Flatten(), nn.Linear(input_dim, output_dim))
        return net

    if isinstance(seq_list, (list, tuple)) and all(isinstance(x, int) for x in seq_list):
        layers = []
        in_dim = input_dim
        for h in seq_list:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))

class TorchDDQN:
    def __init__(self,
                 sequential_list: Union[nn.Sequential, List, Iterable],
                 state_size:int,
                 action_size:int,
                 lr:float = 1e-3,
                 gamma:float = 0.99,
                 buffer_size:int = 100000,
                 batch_size:int = 64,
                 device:Union[str,torch.device]=None,
                 epsilon_start:float=1.0,
                 epsilon_final:float=0.01,
                 epsilon_decay:float=500.0,
                 target_update:int=1000,
                 learn_interval:int=16,
                 tau:float=0.005, 
                 grad_clip:float=10.0, 
                 min_replay_size:int=1000
                 ):
        """
        sequential_list: conforme descrito acima.
        state_size: dimensão do estado (flattened)
        action_size: número de ações (4 no seu ambiente)
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start 
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.target_update = target_update
        self.learn_steps = 0
        self.learn_interval = learn_interval
        self.tau = tau               
        self.grad_clip = grad_clip
        self.min_replay_size = min_replay_size

        self.policy_net = build_network_from_sequential_list(sequential_list, state_size, action_size).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss = 0.0
        self.loss_fn = nn.HuberLoss()

        self.replay = ReplayBuffer(capacity=buffer_size)

    def update_epsilon(self):
        """Atualiza epsilon usando decaimento exponencial."""
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                    np.exp(-self.steps_done / self.epsilon_decay)

    def act(self, state: np.ndarray, eval: bool = False) -> int:
        if not eval:
            self.steps_done += 1

        eps = 0.0 if eval else self.epsilon
        if (not eval) and rand.random() < eps:
            return rand.randrange(self.action_size)
        
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device)
            if s.ndim == 1:
                s = s.unsqueeze(0)
            q = self.policy_net(s)
            return int(torch.argmax(q, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, float(reward), next_state, bool(done))
        self.learn_steps += 1
        if self.learn_steps % self.learn_interval == 0:
            self.learn()

    def learn(self):
        # BUFFER WARMUP
        if len(self.replay) < self.min_replay_size:
            return None

        batch = self.replay.sample(self.batch_size)

        def stack_states(list_of_states):
            arrs = [np.asarray(s, dtype=np.float32).reshape(-1) for s in list_of_states]
            return np.vstack(arrs)
                
        states = stack_states(batch.state)
        next_states = stack_states(batch.next_state)
    
        actions = np.array(batch.action, dtype=np.int64)
        rewards = np.array(batch.reward, dtype=np.float32)
        dones   = np.array(batch.done, dtype=np.uint8)

        states_t = torch.from_numpy(states).float().to(self.device)         # (B, S)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        self.loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        self.loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        
        self.optimizer.step()

        self.learn_steps += 1 

        # POLYAK UPDATES
        if self.tau and self.tau > 0:
            for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)
        else:
            if self.learn_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path_policy:str, path_target:str=None):
        torch.save(self.policy_net.state_dict(), path_policy)
        if path_target:
            torch.save(self.target_net.state_dict(), path_target)

    def load(self, path_policy:str, path_target:str=None):
        self.policy_net.load_state_dict(torch.load(path_policy, map_location=self.device))
        if path_target:
            self.target_net.load_state_dict(torch.load(path_target, map_location=self.device))
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def __call__(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.policy_net(state).detach().cpu()
    
def exp_decay_factor_to(final_epsilon: float, 
                        final_step: int, 
                        epsilon_start: float = 1.0,
                        convergence_threshold: float = 0.01) -> float:
    """
    Calcula o epsilon_decay para atingir final_epsilon no final_step.
    
    Usa a fórmula de decaimento exponencial:
        epsilon(t) = epsilon_final + (epsilon_start - epsilon_final) * exp(-t / epsilon_decay)
    
    Resolve para epsilon_decay tal que:
        epsilon(final_step) ≈ final_epsilon
    
    Args:
        final_epsilon: Valor desejado de epsilon no step final (ex: 0.01)
        final_step: Número do step onde epsilon deve atingir final_epsilon (ex: 5000)
        epsilon_start: Valor inicial de epsilon (padrão: 1.0)
        convergence_threshold: Quão próximo de final_epsilon queremos chegar.
                              0.01 = chega a 99% do caminho (padrão)
                              0.05 = chega a 95% do caminho (mais rápido)
    
    Returns:
        epsilon_decay: Constante de tempo para usar no DDQNAgent
    
    Raises:
        ValueError: Se parâmetros inválidos
    
    """
    # Validações
    if final_step <= 0:
        raise ValueError(f"final_step deve ser > 0, recebido: {final_step}")
    
    if final_epsilon <= 0 or final_epsilon >= epsilon_start:
        raise ValueError(
            f"final_epsilon deve estar entre 0 e epsilon_start. "
            f"Recebido: final_epsilon={final_epsilon}, epsilon_start={epsilon_start}"
        )
    
    if not (0 < convergence_threshold < 1):
        raise ValueError(
            f"convergence_threshold deve estar entre 0 e 1, recebido: {convergence_threshold}"
        )
    
    epsilon_decay = -final_step / np.log(convergence_threshold)
    
    return epsilon_decay

class SAECollabDDQN:
    
    def __init__(self,
                 state_size:int,
                 action_size:int,
                 # HIPERPARAMETROS SAECOLLAB
                 first_hidden_size: int,
                 hidden_activation: Optional[nn.Module] = None,
                 out_activation: Optional[nn.Module] = None,
                 accelerate_etas: bool = False,
                 # HIPERPARAMETROS RL
                 lr:Union[float,List[float]] = 1e-3,
                 gamma:float = 0.99,
                 buffer_size:int = 100000,
                 batch_size:int = 64,
                 epsilon_start:float=1.0,
                 device:Union[str,torch.device]=None,
                 epsilon_final:float=0.01,
                 epsilon_decay:float=500.0,
                 target_update:int=1000,
                 learn_interval:int=16,
                 tau:float=0.005, 
                 grad_clip:float=10.0, 
                 min_replay_size:int=1000
                 ):
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_size = state_size
        self.action_size = action_size
        
        # BUILDING NETS
        self.policy_net = SAECollabNet(state_size,
                                       first_hidden_size,
                                       action_size,
                                       hidden_activation=hidden_activation,
                                       out_activation=out_activation,
                                       accelerate_etas=accelerate_etas,
                                       device=self.device
                                       )
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)        
        self.target_net.eval()
        
        # BACKWARD RELATED
        self.lr        = lr
        policy_params  = [p for p in self.policy_net.parameters() if p.requires_grad]
        _lr = self.lr if isinstance(self.lr,float) else self.lr[0]
        self.optimizer = optim.Adam(policy_params, lr=_lr)
        self.loss_fn   = nn.HuberLoss()
        self.loss      = 0.0 

        self.layers_added    = 0 
        self.steps_done      = 0
        self.learn_steps     = 0
        
        self.gamma           = gamma
        self.batch_size      = batch_size
        self.epsilon_start   = epsilon_start
        self.epsilon         = epsilon_start
        self.epsilon_final   = epsilon_final
        self.epsilon_decay   = epsilon_decay
        self.target_update   = target_update
        self.learn_interval  = learn_interval
        self.tau             = tau
        self.grad_clip       = grad_clip
        self.min_replay_size = min_replay_size

        self.replay = ReplayBuffer(capacity=buffer_size)
    
    def update_epsilon(self):
        """Atualiza epsilon usando decaimento exponencial."""
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                    np.exp(-self.steps_done / self.epsilon_decay)
    
    def act(self, state: np.ndarray, eval: bool = False) -> int:
        if not eval:
            self.steps_done += 1
                    
        eps = 0.0 if eval else self.epsilon
        if (not eval) and rand.random() < eps:
            return rand.randrange(self.action_size)
        
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device)
            if s.ndim == 1:
                s = s.unsqueeze(0)
            q, _, _ = self.policy_net(s)
            return int(torch.argmax(q, dim=1).item())
        
    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, float(reward), next_state, bool(done))
        self.learn_steps += 1
        if self.learn_steps % self.learn_interval == 0:
            self.learn()

    def save(self, path_policy:str, path_target:str=None):
        torch.save(self.policy_net, path_policy)
        if path_target:
            torch.save(self.target_net, path_target)
    
    def add_layer(self,
                  layer_hidden_size,
                  layer_extra_size: Optional[int] = None,
                  k: float = 1.0,
                  mutation_mode: Optional[str] = None,
                  target_fn: Optional[nn.Module] = None,
                  eta: float = 0.0,
                  eta_increment: float = 0.001,
                  hidden_activation: Optional[nn.Module] = None,
                  out_activation: Optional[nn.Module] = None,
                  extra_activation: Optional[nn.Module] = None,
                  accelerate_factor: float = 2.0,
                  is_k_trainable=True
                  ):
        
        self.policy_net.add_layer(
            layer_hidden_size,
            self.action_size,
            extra_dim         = layer_extra_size,
            k                 = k,
            mutation_mode     = mutation_mode,
            target_fn         = target_fn,
            eta               = eta,
            eta_increment     = eta_increment,
            hidden_activation = hidden_activation,
            out_activation    = out_activation,
            extra_activation  = extra_activation,
            accelerate_factor = accelerate_factor,
            is_k_trainable    = is_k_trainable
        )

        self.layers_added += 1
        _lr = self.lr if isinstance(self.lr,float) else \
                         self.lr[self.layers_added] if self.layers_added < len(self.lr) else \
                         self.lr[-1]
        policy_params  = [p for p in self.policy_net.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(policy_params, lr=_lr)

    def learn(self):
        # BUFFER WARMUP
        if len(self.replay) < self.min_replay_size:
            return None

        batch = self.replay.sample(self.batch_size)

        def stack_states(list_of_states):
            arrs = [np.asarray(s, dtype=np.float32).reshape(-1) for s in list_of_states]
            return np.vstack(arrs)
                
        states = stack_states(batch.state)
        next_states = stack_states(batch.next_state)
    
        actions = np.array(batch.action, dtype=np.int64)
        rewards = np.array(batch.reward, dtype=np.float32)
        dones   = np.array(batch.done, dtype=np.uint8)

        states_t = torch.from_numpy(states).float().to(self.device)         # (B, S)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        q_values_all, _, _ = self.policy_net(states_t)
        q_values = q_values_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values_all, _, _ = self.policy_net(next_states_t)
            next_actions = next_q_values_all.argmax(1)
            
            target_next_q_all, _, _ = self.target_net(next_states_t)
            next_q = target_next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        self.loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        self.loss.backward()
        # TODO: REFERENCIAR O VALOR USADO NO GRAD CLIPING
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()
        self.learn_steps += 1 

        # POLYAK UPDATE
        if len(self.policy_net.layers) != len(self.target_net.layers):
            self.target_net = copy.deepcopy(self.policy_net).to(self.device)        
            self.target_net.eval()
        if self.tau and self.tau > 0:
            for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)
        else:
            if self.learn_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


    def load(self, path_policy:str, path_target:str=None):
        self.policy_net = torch.load(path_policy, map_location=self.device,weights_only=False)
        if path_target:
            self.target_net = torch.load(path_target, map_location=self.device,weights_only=False)
        else:
            self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

    def __call__(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # Ensure state has batch dimension
        if state.ndim == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # FIX: Unpack tuple and return only q-values
        q_values, _, _ = self.policy_net(state)
        
        # Remove batch dimension if input was single state
        if squeeze_output:
            q_values = q_values.squeeze(0)
        
        return q_values.detach().cpu()