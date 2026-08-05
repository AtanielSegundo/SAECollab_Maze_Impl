#!python3 ablation/config.py

from typing import *
from dataclasses import dataclass
from enum import Enum

from StackedCollab.collabNet import LayersConfig
from env.MazeWrapper import StateEncoder


@dataclass
class GlobalHyperparameters:
    learning_rate:float
    discount_factor:float
    epsilon_decay:float
    episodes:int
    max_steps:int
    batch_size:int
    steps_learn_interval:int
    rolling_window_size:int
    new_layer_learning_rate:float
    insert_patience:int
    insert_min_goals:int
    insert_min_variance:float

    # ── Optional convergence-improvement features ─────────────────────────
    # Potential-based reward shaping (Ng et al. 1999).
    # Adds F(s, s') = gamma * Phi(s') - Phi(s) with Phi based on Manhattan
    # distance to goal. Preserves the optimal policy.
    use_reward_shaping: bool  = False

    # Prioritized Experience Replay (Schaul et al. 2016, proportional variant).
    use_per           : bool  = False
    per_alpha         : float = 0.6     # priority exponent (0 = uniform)
    per_beta_start    : float = 0.4     # IS-correction exponent at training start
    per_beta_final    : float = 1.0     # IS-correction exponent at end (annealed)
    per_beta_steps    : int   = 100_000 # steps over which beta is annealed
    per_eps           : float = 1e-6    # added to |TD-error| before exponentiation

class LayerInsertionType(Enum):
    CNT = "CNT"
    CRT = "CRT"
    DRT = "DRT"
    ALT = "ALT"

    @property
    def tag(self):
        return self.value

class LayerMode:
    def __init__(self,
                 use_extra_branch:bool,
                 is_k_trainable:bool
                ):
        self.is_k_trainable   = is_k_trainable
        self.use_extra_branch = use_extra_branch

class LayerModeType(Enum):
    M1 = LayerMode(False, False)
    M2 = LayerMode(False, True)
    M3 = LayerMode(True , False)
    M4 = LayerMode(True , True)

    @property
    def tag(self):
        return str(self).split(".")[-1]

    @staticmethod
    def from_tag(tag:str) -> Self:
        if tag == "M1":
            return LayerModeType.M1
        elif tag == "M2":
            return LayerModeType.M2
        elif tag == "M3":
            return LayerModeType.M3
        elif tag == "M4":
            return LayerModeType.M4

class ModelArch:
    def __init__(self,
                 max_layers             : int,
                 width_multipliers      : LayersConfig,   # THE OUT LAYER MULTIPLIER SHOULD BE 1
                 delta_width_multipliers: LayersConfig,
                 activation             : LayersConfig,
                 use_bias               : LayersConfig,
                 is_static              : bool = False,   # Static Architectures have an static number of neurons 
                                                          # The default case is an factor to multiply by an extern
                                                          # number of neurons 
                 )         :
        self.max_layers              = max_layers
        self.width_multipliers       = width_multipliers
        self.delta_width_multipliers = delta_width_multipliers
        self.activation              = activation
        self.use_bias                = use_bias
        self.is_static               = is_static

    @property
    def tag(self):
        def GAN(act) -> str:
            #GET ACTIVATION NAME
            return str(act).split(".")[-1].split("'")[0]
        def GUB(b:bool) -> str:
            return "T" if b else "F"

        return f"{self.max_layers}" \
               f"_H{self.width_multipliers.hidden}O{self.width_multipliers.out}E{self.width_multipliers.extra}" \
               f"_H{GAN(self.activation.hidden)}O{GAN(self.activation.out)}E{GAN(self.activation.extra)}" \
               f"_H{GUB(self.use_bias.hidden)}O{GUB(self.use_bias.out)}E{GUB(self.use_bias.extra)}"


class StateRepresentation:
    NUM_LAST_STATES_TAG          = "nls"
    NUM_LAST_ACTIONS_TAG         = "nla"
    POSSIBLE_ACTIONS_FEATURE_TAG = "paf"
    VISITED_COUNT_TAG            = "vcnt"

    def __init__(self,
        state_encoder           :StateEncoder  = StateEncoder.COORDS,
        num_last_states         :int           = None,
        num_last_actions        :int           = None,
        possible_actions_feature:bool          = False,
        visited_count           :bool          = False,
    ) :
        self.opts = {
            "state_encoder"           : state_encoder,
            "num_last_states"         : num_last_states,
            "num_last_actions"        : num_last_actions,
            "possible_actions_feature": possible_actions_feature,
            "visited_count"           : visited_count
        }

    @property
    def tag(self):
        tag = ""
        tag += str(self.opts["state_encoder"]).split(".")[-1]
        if not self.opts["num_last_states"] is None:
            tag += f"_{StateRepresentation.NUM_LAST_STATES_TAG}_" + str(self.opts["num_last_states"])
        if not self.opts["num_last_actions"] is None:
            tag += f"_{StateRepresentation.NUM_LAST_ACTIONS_TAG}_" + str(self.opts["num_last_actions"])
        if self.opts["possible_actions_feature"]:
            tag += f"_{StateRepresentation.POSSIBLE_ACTIONS_FEATURE_TAG}"
        if self.opts["visited_count"]:
            tag += f"_{StateRepresentation.VISITED_COUNT_TAG}"
        return tag
