import torch

torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

import torch.nn as nn
import torch.optim as optim
import numpy as np
import random as rand
import copy

from torch.amp import autocast, GradScaler
from collections import deque, namedtuple
from typing import *

from StackedCollab.collabNet import ReservedSAECollabNet, SAECollabNet, LayersConfig, \
                                    MutationMode, NewLayerCfg

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(
            torch.as_tensor(state, dtype=torch.float32),
            action,
            reward,
            torch.as_tensor(next_state, dtype=torch.float32),
            done
        ))

    def sample(self, batch_size: int):
        batch = rand.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.buffer)

class FastReplayBuffer:
    """
    Preallocated ring buffer for states/next_states/actions/rewards/dones.
    Stores tensors (pinned CPU or GPU). sample(batch_size, device, non_blocking=True)
    returns tensors already on the requested device.
    """
    def __init__(self, capacity: int, state_shape: Tuple[int, ...],
                 storage_device: Union[str, torch.device] = "cpu",
                 pin_memory: bool = True, dtype=torch.float32):
        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape)
        self.idx = 0
        self.size = 0
        self.dtype = dtype

        self.storage_device = torch.device(storage_device)
        self.pin_memory = bool(pin_memory) and (self.storage_device.type == "cpu")

        alloc_kwargs = {"dtype": self.dtype}
        if self.pin_memory:
            alloc_kwargs["pin_memory"] = True

        # allocate storage tensors
        self.states = torch.empty((self.capacity, *self.state_shape), **alloc_kwargs)
        self.next_states = torch.empty((self.capacity, *self.state_shape), **alloc_kwargs)
        self.actions = torch.empty((self.capacity,), dtype=torch.long)
        self.rewards = torch.empty((self.capacity,), dtype=torch.float32)
        self.dones = torch.empty((self.capacity,), dtype=torch.uint8)

        # move storage to GPU if requested
        if self.storage_device.type == "cuda":
            self.states = self.states.to(self.storage_device)
            self.next_states = self.next_states.to(self.storage_device)
            self.actions = self.actions.to(self.storage_device)
            self.rewards = self.rewards.to(self.storage_device)
            self.dones = self.dones.to(self.storage_device)

    def push(self, state, action, reward, next_state, done):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=self.dtype)
        if not torch.is_tensor(next_state):
            next_state = torch.as_tensor(next_state, dtype=self.dtype)

        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        if next_state.ndim == 2 and next_state.shape[0] == 1:
            next_state = next_state.squeeze(0)

        if state.device != self.states.device:
            state = state.to(self.states.device)
        if next_state.device != self.next_states.device:
            next_state = next_state.to(self.next_states.device)

        self.states[self.idx].copy_(state.reshape(self.state_shape))
        self.next_states[self.idx].copy_(next_state.reshape(self.state_shape))
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = 1 if done else 0

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return int(self.size)

    def sample(self, batch_size: int, device: Optional[Union[str, torch.device]] = None,
               non_blocking: bool = True):
        assert self.size > 0, "Buffer is empty"
        # sample indices on storage device (fast)
        idxs = torch.randint(0, self.size, (batch_size,), dtype=torch.long, device=self.states.device)

        states_b = self.states.index_select(0, idxs)
        next_states_b = self.next_states.index_select(0, idxs)
        actions_b = self.actions.index_select(0, idxs)
        rewards_b = self.rewards.index_select(0, idxs)
        dones_b = self.dones.index_select(0, idxs)

        if device is None:
            return states_b, actions_b, rewards_b, next_states_b, dones_b

        target_device = torch.device(device)
        if target_device != self.states.device:
            states_b = states_b.to(target_device, non_blocking=non_blocking)
            next_states_b = next_states_b.to(target_device, non_blocking=non_blocking)
            actions_b = actions_b.to(target_device, non_blocking=non_blocking)
            rewards_b = rewards_b.to(target_device, non_blocking=non_blocking)
            dones_b = dones_b.to(target_device, non_blocking=non_blocking)

        return states_b, actions_b, rewards_b, next_states_b, dones_b

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
        final_step: Número do step onde epsilon deve atingir final_epsilon  (ex: 5000)
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

def build_network_from_sequential_list(seq_list, input_dim: int, output_dim: int) -> nn.Module:
    """Mantido exatamente igual (não afeta performance)"""
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

    if isinstance(seq_list, (list, tuple)) and len(seq_list) > 0 and all(isinstance(m, nn.Module) for m in seq_list):
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
                 sequential_list,
                 state_size: int,
                 action_size: int,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 device=None,
                 epsilon_start: float = 1.0,
                 epsilon_final: float = 0.01,
                 epsilon_decay: float = 500.0,
                 target_update: int = 1000,
                 learn_interval: int = 16,
                 tau: float = 0.005,
                 grad_clip: float = 10.0,
                 min_replay_size: int = 1000):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
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
        self.loss = 0.0  # FIX: initialise so hasattr() always finds it

        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.policy_net = build_network_from_sequential_list(sequential_list, state_size, action_size).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.HuberLoss()

        # self.replay = ReplayBuffer(capacity=buffer_size)
        self.replay = FastReplayBuffer(capacity=buffer_size,
                                        state_shape=(self.state_size,),
                                        storage_device="cpu",
                                        pin_memory=torch.cuda.is_available(),
                                        dtype=torch.float32
                                        )


    def compile(self):
        if torch.__version__ >= "2.0" and self.device.type == "cuda":
            try:
                self.policy_net = torch.compile(self.policy_net, mode="reduce-overhead")
                self.target_net = torch.compile(self.target_net, mode="reduce-overhead")
            except Exception:
                pass

    def update_epsilon(self):
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                       np.exp(-self.steps_done / self.epsilon_decay)

    def act(self, state, eval: bool = False) -> int:
        if not eval:
            self.steps_done += 1

        if not eval and rand.random() < self.epsilon:
            return rand.randrange(self.action_size)

        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                s = state if state.device == self.device else state.to(self.device, non_blocking=True)
            else:
                s = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device, non_blocking=True)
            if s.ndim == 1:
                s = s.unsqueeze(0)
            q = self.policy_net(s)
            return int(torch.argmax(q, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
        self.learn_steps += 1  # FIX: only incremented here, never inside learn()
        if self.learn_steps % self.learn_interval == 0:
            self.learn()

    def learn(self):
        if len(self.replay) < self.min_replay_size:
            return

        states_t, actions_t, rewards_t, next_states_t, dones_t = \
        self.replay.sample(self.batch_size, device=self.device, non_blocking=True)

        # ensure dtypes
        actions_t = actions_t.long()
        rewards_t = rewards_t.float()
        dones_t = dones_t.float()

        if self.use_amp:
            # FIX: only the forward passes run under autocast (fp16).
            # The loss is computed in fp32 OUTSIDE autocast to prevent
            # fp16 underflow (FTZ) turning small initial losses into 0.0.
            with autocast("cuda"):
                q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = self.policy_net(next_states_t).argmax(1)
                    next_q       = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q     = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            # fp32 loss — outside autocast
            loss = self.loss_fn(q_values.float(), target_q.float())
            self.loss = loss.item()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.policy_net(next_states_t).argmax(1)
                next_q       = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q     = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            loss = self.loss_fn(q_values, target_q)
            self.loss = loss.item()  # FIX: store scalar, not tensor

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
            self.optimizer.step()

        # Polyak / hard update
        if self.tau and self.tau > 0:
            for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)
        elif self.learn_steps % self.target_update == 0:
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

    def __call__(self, state, return_numpy: bool = False):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            if state.device != self.device:
                state = state.to(self.device, non_blocking=True)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        out = self.policy_net(state).detach()
        if return_numpy:
            return out.cpu().numpy()
        return out
    

class SAECollabDDQN:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 first_hidden_size: int,
                 hidden_activation: Optional[nn.Module] = None,
                 out_activation: Optional[nn.Module] = None,
                 accelerate_etas: bool = False,
                 lr: Union[float, List[float]] = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 epsilon_start: float = 1.0,
                 device=None,
                 epsilon_final: float = 0.01,
                 epsilon_decay: float = 500.0,
                 target_update: int = 1000,
                 learn_interval: int = 16,
                 tau: float = 0.005,
                 grad_clip: float = 10.0,
                 min_replay_size: int = 1000,
                 use_bias: LayersConfig = None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            
        self.state_size = state_size
        self.action_size = action_size
        self.loss = 0.0  # FIX: initialise so hasattr() always finds it

        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        use_bias = use_bias or LayersConfig(True, True, True)

        self.policy_net = SAECollabNet(state_size, first_hidden_size, action_size,
                                       hidden_activation=hidden_activation,
                                       out_activation=out_activation,
                                       accelerate_etas=accelerate_etas,
                                       device=self.device,
                                       use_bias=use_bias)
                                       
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.lr = lr
        policy_params = [p for p in self.policy_net.parameters() if p.requires_grad]
        _lr = self.lr if isinstance(self.lr, float) else self.lr[0]
        self.optimizer = optim.Adam(policy_params, lr=_lr)
        self.loss_fn = nn.HuberLoss()

        self.layers_added = 0
        self.steps_done = 0
        self.learn_steps = 0

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.learn_interval = learn_interval
        self.tau = tau
        self.grad_clip = grad_clip
        self.min_replay_size = min_replay_size

        # self.replay = ReplayBuffer(capacity=buffer_size)
        self.replay = FastReplayBuffer(capacity=buffer_size,
                                        state_shape=(self.state_size,),
                                        storage_device="cpu",
                                        pin_memory=torch.cuda.is_available(),
                                        dtype=torch.float32
                                        )

    def compile(self):
        if torch.__version__ >= "2.0" and self.device.type == "cuda":
            try:
                self.policy_net = torch.compile(self.policy_net, mode="reduce-overhead")
                self.target_net = torch.compile(self.target_net, mode="reduce-overhead")
            except Exception:
                pass

    def update_epsilon(self):
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                       np.exp(-self.steps_done / self.epsilon_decay)

    def act(self, state, eval: bool = False) -> int:
        if not eval:
            self.steps_done += 1
        
        if not eval and rand.random() < self.epsilon:
            return rand.randrange(self.action_size)

        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                s = state if state.device == self.device else state.to(self.device, non_blocking=True)
            else:
                s = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device, non_blocking=True)
            
            if s.ndim == 1:
                s = s.unsqueeze(0)
            
            q, _, _ = self.policy_net(s)
            return int(torch.argmax(q, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
        self.learn_steps += 1  # FIX: only incremented here, never inside learn()
        if self.learn_steps % self.learn_interval == 0:
            self.learn()

    def add_layer(self, layer_hidden_size, layer_extra_size=None, k=1.0, mutation_mode=None,
                  target_fn=None, eta=0.0, eta_increment=0.001, hidden_activation=None,
                  out_activation=None, extra_activation=None, accelerate_factor=2.0,
                  is_k_trainable=True, use_bias=None):
        """Atualiza AMBOS os networks (sem deepcopy no learn)"""
        self.policy_net.add_layer(
            layer_hidden_size, self.action_size, extra_dim=layer_extra_size, k=k,
            mutation_mode=mutation_mode, target_fn=target_fn, eta=eta,
            eta_increment=eta_increment, hidden_activation=hidden_activation,
            out_activation=out_activation, extra_activation=extra_activation,
            accelerate_factor=accelerate_factor, is_k_trainable=is_k_trainable,
            use_bias=use_bias
        )
        self.target_net.add_layer(
            layer_hidden_size, self.action_size, extra_dim=layer_extra_size, k=k,
            mutation_mode=mutation_mode, target_fn=target_fn, eta=eta,
            eta_increment=eta_increment, hidden_activation=hidden_activation,
            out_activation=out_activation, extra_activation=extra_activation,
            accelerate_factor=accelerate_factor, is_k_trainable=is_k_trainable,
            use_bias=use_bias
        )

        self.layers_added += 1
        _lr = self.lr if isinstance(self.lr, float) else \
              self.lr[self.layers_added] if self.layers_added < len(self.lr) else self.lr[-1]
        policy_params = [p for p in self.policy_net.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(policy_params, lr=_lr)

    def learn(self):
        if len(self.replay) < self.min_replay_size:
            return

        states_t, actions_t, rewards_t, next_states_t, dones_t = \
        self.replay.sample(self.batch_size, device=self.device, non_blocking=True)

        # ensure dtypes
        actions_t = actions_t.long()
        rewards_t = rewards_t.float()
        dones_t = dones_t.float()
        
        if self.use_amp:
            # FIX: only forward passes under autocast; loss computed in fp32
            # outside the context to prevent fp16 FTZ underflow → 0.0 loss.
            with autocast("cuda"):
                q_values_all, _, _ = self.policy_net(states_t)
                q_values = q_values_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values_all, _, _ = self.policy_net(next_states_t)
                    next_actions = next_q_values_all.argmax(1)
                    target_next_q_all, _, _ = self.target_net(next_states_t)
                    next_q   = target_next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            # fp32 loss — outside autocast
            loss = self.loss_fn(q_values.float(), target_q.float())
            self.loss = loss.item()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            q_values_all, _, _ = self.policy_net(states_t)
            q_values = q_values_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values_all, _, _ = self.policy_net(next_states_t)
                next_actions = next_q_values_all.argmax(1)
                target_next_q_all, _, _ = self.target_net(next_states_t)
                next_q   = target_next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            loss = self.loss_fn(q_values, target_q)
            self.loss = loss.item()  # FIX: store scalar, not tensor

            self.optimizer.zero_grad()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # FIX: learn_steps is NOT incremented here — only in remember()

        # Polyak (ou hard update)
        if self.tau and self.tau > 0:
            for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)
        elif self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path_policy:str, path_target:str=None):
        torch.save(self.policy_net, path_policy)
        if path_target:
            torch.save(self.target_net, path_target)
    
    def load(self, path_policy:str, path_target:str=None):
        self.policy_net = torch.load(path_policy, map_location=self.device, weights_only=False)
        if path_target:
            self.target_net = torch.load(path_target, map_location=self.device, weights_only=False)
        else:
            self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()
    
    def __call__(self, state, return_numpy: bool = False):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            if state.device != self.device:
                state = state.to(self.device, non_blocking=True)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        out = self.policy_net(state).detach()
        if return_numpy:
            return out.cpu().numpy()
        return out
    
class ReservedSAECollabDDQN:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 reserved_layers_cfg: List[NewLayerCfg],
                 accelerate_etas: bool = False,
                 accelerate_factor: float = 2.0,
                 lr: Union[float, List[float]] = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 epsilon_start: float = 1.0,
                 device=None,
                 epsilon_final: float = 0.01,
                 epsilon_decay: float = 500.0,
                 target_update: int = 1000,
                 learn_interval: int = 16,
                 tau: float = 0.005,
                 grad_clip: float = 10.0,
                 min_replay_size: int = 1000,
                 use_bias: LayersConfig = None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.loss = 0.0  # FIX: initialise so hasattr() always finds it

        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        use_bias = use_bias or LayersConfig(True, True, True)

        self.policy_net = ReservedSAECollabNet(state_size, reserved_layers_cfg,
                                               device=self.device,
                                               accelerate_etas=accelerate_etas,
                                               accelerate_factor=accelerate_factor)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        if isinstance(lr, float):
            self.lr = [lr] * len(reserved_layers_cfg)
        else:
            self.lr = lr + [lr[-1]] * (len(reserved_layers_cfg) - len(lr)) if len(lr) < len(reserved_layers_cfg) else lr[:len(reserved_layers_cfg)]

        self.optimizer = optim.Adam([{'params': layer.parameters(), 'lr': self.lr[i]}
                                     for i, layer in enumerate(self.policy_net.layers)])
        self.set_optimizer_to_head()

        self.loss_fn = nn.HuberLoss()

        self.layers_added = 0
        self.steps_done = 0
        self.learn_steps = 0

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.learn_interval = learn_interval
        self.tau = tau
        self.grad_clip = grad_clip
        self.min_replay_size = min_replay_size

        # self.replay = ReplayBuffer(capacity=buffer_size)
        self.replay = FastReplayBuffer(capacity=buffer_size,
                                        state_shape=(self.state_size,),
                                        storage_device="cpu",
                                        pin_memory=torch.cuda.is_available(),
                                        dtype=torch.float32
                                        )

    def compile(self):
        if torch.__version__ >= "2.0" and self.device.type == "cuda":
            try:
                self.policy_net = torch.compile(self.policy_net, mode="reduce-overhead")
                self.target_net = torch.compile(self.target_net, mode="reduce-overhead")
            except Exception:
                pass

    def set_optimizer_to_head(self):
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.lr[i] if i == self.policy_net.active_head else 0.0

    def update_epsilon(self):
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                       np.exp(-self.steps_done / self.epsilon_decay)

    def act(self, state, eval: bool = False) -> int:
        if not eval:
            self.steps_done += 1
        
        if not eval and rand.random() < self.epsilon:
            return rand.randrange(self.action_size)

        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                s = state if state.device == self.device else state.to(self.device, non_blocking=True)
            else:
                s = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device, non_blocking=True)
            
            if s.ndim == 1:
                s = s.unsqueeze(0)
            
            q, _, _ = self.policy_net(s)
            return int(torch.argmax(q, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
        self.learn_steps += 1  # FIX: only incremented here, never inside learn()
        if self.learn_steps % self.learn_interval == 0:
            self.learn()

    def use_next_layer(self):
        self.policy_net.use_next_layer()
        self.target_net.use_next_layer()
        self.layers_added += 1
        self.set_optimizer_to_head()

    def learn(self):
        if len(self.replay) < self.min_replay_size:
            return

        states_t, actions_t, rewards_t, next_states_t, dones_t = \
        self.replay.sample(self.batch_size, device=self.device, non_blocking=True)

        # ensure dtypes
        actions_t = actions_t.long()
        rewards_t = rewards_t.float()
        dones_t = dones_t.float()

        if self.use_amp:
            # FIX: only forward passes under autocast; loss computed in fp32
            # outside the context to prevent fp16 FTZ underflow → 0.0 loss.
            with autocast("cuda"):
                q_values_all, _, _ = self.policy_net(states_t)
                q_values = q_values_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values_all, _, _ = self.policy_net(next_states_t)
                    next_actions = next_q_values_all.argmax(1)
                    target_next_q_all, _, _ = self.target_net(next_states_t)
                    next_q   = target_next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            # fp32 loss — outside autocast
            loss = self.loss_fn(q_values.float(), target_q.float())
            self.loss = loss.item()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            q_values_all, _, _ = self.policy_net(states_t)
            q_values = q_values_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values_all, _, _ = self.policy_net(next_states_t)
                next_actions = next_q_values_all.argmax(1)
                target_next_q_all, _, _ = self.target_net(next_states_t)
                next_q   = target_next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            loss = self.loss_fn(q_values, target_q)
            self.loss = loss.item()  # FIX: store scalar, not tensor

            self.optimizer.zero_grad()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # FIX: learn_steps is NOT incremented here — only in remember()

        # Polyak apenas nas camadas ativas
        if self.tau and self.tau > 0:
            active_head = self.policy_net.active_head
            policy_params = [p for layer in self.policy_net.layers[:active_head + 1] for p in layer.parameters()]
            target_params = [p for layer in self.target_net.layers[:active_head + 1] for p in layer.parameters()]
            with torch.no_grad():
                for p, tp in zip(policy_params, target_params):
                    tp.data.mul_(1.0 - self.tau)
                    tp.data.add_(self.tau * p.data)
        elif self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path_policy:str, path_target:str=None):
        torch.save(self.policy_net, path_policy)
        if path_target:
            torch.save(self.target_net, path_target)
    
    def load(self, path_policy:str, path_target:str=None):
        self.policy_net = torch.load(path_policy, map_location=self.device, weights_only=False)
        if path_target:
            self.target_net = torch.load(path_target, map_location=self.device, weights_only=False)
        else:
            self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()
    
    def __call__(self, state, return_numpy: bool = False):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            if state.device != self.device:
                state = state.to(self.device, non_blocking=True)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        out = self.policy_net(state).detach()
        if return_numpy:
            return out.cpu().numpy()
        return out