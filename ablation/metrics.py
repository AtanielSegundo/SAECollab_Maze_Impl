#!python3 ablation/metrics.py

import csv
import os
import math
from typing import *
from dataclasses import dataclass, field, fields


def _derive_branch_inserted_in(existing: List[int], branchs: List[int]) -> List[int]:
    """If CSV already has branch_inserted_in column (non-empty and non-zero), trust it.
    Otherwise derive as positive diffs of branchs count per episode."""
    if existing and any(v for v in existing):
        return list(existing)
    if not branchs:
        return [0] * len(existing)
    derived = [0] * len(branchs)
    prev = 0
    for i, b in enumerate(branchs):
        derived[i] = 1 if b > prev else 0
        prev = b
    return derived


@dataclass
class ModelTrainMetrics:
    episode         : List[int]   = field(default_factory=list)
    reward          : List[float] = field(default_factory=list)
    cumulative_goals: List[int]   = field(default_factory=list)
    success_rate    : List[float] = field(default_factory=list)
    loss            : List[float] = field(default_factory=list)
    steps           : List[int]   = field(default_factory=list)
    parameters      : List[int]   = field(default_factory=list)
    delta_time      : List[float] = field(default_factory=list)
    branchs         : List[int]   = field(default_factory=list)
    branch_inserted_in: List[int] = field(default_factory=list)

    type_map = {
        'episode'           : int,
        'cumulative_goals'  : int,
        'steps'             : int,
        'parameters'        : int,
        'reward'            : float,
        'success_rate'      : float,
        'loss'              : float,
        'delta_time'        : float,
        'branchs'           : int,
        'branch_inserted_in': int,
    }

    header_map = {
        'episode'         : 'episode',
        'reward'          : 'reward',

        'cumulative_goals': 'cumulative_goals',
        'cum_goals'       : 'cumulative_goals',

        'success_rate'    : 'success_rate',
        'sucess_rate'     : 'success_rate',                          # aceita a grafia errada caso exista

        'training_loss'   : 'loss',
        'loss'            : 'loss',
        'steps'           : 'steps',

        'parameters'      : 'parameters',
        'params'          : 'parameters',

        'delta_time'      : 'delta_time',

        'branchs'         : 'branchs',

        'branch_inserted_in': 'branch_inserted_in',
    }


    def __post_init__(self):
        self.ordered_metrics = [
            self.episode,
            self.reward,
            self.cumulative_goals,
            self.success_rate,
            self.loss,
            self.steps,
            self.parameters,
            self.delta_time,
            self.branchs,
            self.branch_inserted_in,
        ]

        self.available_metrics = [f.name for f in fields(self.__class__)
                                  if getattr(f, "default_factory", None) is list]

    def __len__(self):
        return len(self.episode)

    def append(self, *args):
        for idx in range(min(len(args), len(self.ordered_metrics))):
            self.ordered_metrics[idx].append(args[idx])

    def save(self, path: str):
        n = len(self)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.available_metrics)
            for idx in range(n):
                row = [metric[idx] if idx < len(metric) else 0
                       for metric in self.ordered_metrics]
                writer.writerow(row)

    @classmethod
    def load(cls, path: str) -> 'ModelTrainMetrics':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metrics file not found: {path}")

        data = {name: [] for name in cls.header_map.values()}

        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV file has no header")

            # normalizar headers do CSV para nomes internos
            csv_to_internal = {}
            for h in reader.fieldnames:
                key = h.strip()
                if key in cls.header_map:
                    csv_to_internal[key] = cls.header_map[key]
                else:
                    # tentar minusculas sem espacos
                    k2 = key.lower().replace(" ", "_")
                    if k2 in cls.header_map:
                        csv_to_internal[key] = cls.header_map[k2]

            def is_float(s:str):
                try:
                    float(s)
                    return True
                except ValueError:
                    return False

            for row_idx, row in enumerate(reader, start=1):
                try:
                    for csv_h, internal in csv_to_internal.items():
                        raw = row.get(csv_h, "").strip()
                        if is_float(raw):
                            val = float(raw)
                            if not math.isfinite(val):
                                val = cls.type_map.get(internal, float)()
                        elif raw.isnumeric():
                            val = int(raw)
                        else:
                            val = cls.type_map.get(internal, float)()
                        data[internal].append(val)

                except ValueError as e:
                    raise ValueError(f"Invalid data format in CSV at line {row_idx}: {e}")

        # garantir que todos os vetores tenham o mesmo comprimento (preencher com zeros se necessario)
        lengths = [len(v) for v in data.values()]
        max_len = max(lengths) if lengths else 0

        for k, v in data.items():
            if len(v) < max_len:
                fill_type = cls.type_map.get(k, float)
                fill_value = fill_type()
                v.extend([fill_value] * (max_len - len(v)))

        return cls(
            episode=data['episode'],
            reward=data['reward'],
            cumulative_goals=data['cumulative_goals'],
            success_rate=data['success_rate'],
            loss=data['loss'],
            steps=data['steps'],
            parameters=data['parameters'],

            delta_time=data.get('delta_time', []),
            branchs=data.get('branchs',[]),
            branch_inserted_in=_derive_branch_inserted_in(
                data.get('branch_inserted_in', []),
                data.get('branchs', []),
            ),
        )

    def __str__(self):
        """Compact summary."""
        n = len(self)
        if n == 0:
            return "ModelTrainMetrics(empty)"

        def last(x): return x[-1] if x else None
        def mean(x): return sum(x) / len(x) if x else math.nan

        return (
            "ModelTrainMetrics\n"
            f"  samples          : {n}\n"
            f"  last episode     : {last(self.episode)}\n"
            f"  last reward      : {last(self.reward):.4f}\n"
            f"  last success rate: {last(self.success_rate):.4f}\n"
            f"  last loss        : {last(self.loss):.6f}\n"
            f"  avg reward       : {mean(self.reward):.4f}\n"
            f"  avg success rate : {mean(self.success_rate):.4f}\n"
        )

    def pretty_print(self, last_n: int = 10):
        """Tabular view of the last N entries."""
        n = len(self)
        if n == 0:
            print("ModelTrainMetrics(empty)")
            return

        start = max(0, n - last_n)
        headers = ["ep", "reward", "cum_goals", "succ_rate", "loss", "steps", "params", "dt"]
        rows = []

        for i in range(start, n):
            rows.append([
                self.episode[i],
                f"{self.reward[i]:.4f}",
                self.cumulative_goals[i],
                f"{self.success_rate[i]:.4f}",
                f"{self.loss[i]:.6f}",
                self.steps[i],
                self.parameters[i],
                f"{self.delta_time[i]:.3f}" if i < len(self.delta_time) else "",
            ])

        # calcular larguras das colunas (inclui cabecalho)
        cols = list(zip(*([headers] + rows)))
        widths = [max(len(str(v)) for v in col) for col in cols]

        def fmt(row):
            return " | ".join(str(v).rjust(w) for v, w in zip(row, widths))

        sep = "-+-".join("-" * w for w in widths)

        print(fmt(headers))
        print(sep)
        for r in rows:
            print(fmt(r))


@dataclass
class TrainTargetClosure:
    fn:Callable[[str,str,str],str]
    tag:str
    save_model_path:str
    save_metrics_path:str
