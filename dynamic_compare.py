from typing import *

import os
import sys
import csv
import json
import time
import argparse
import subprocess
import statistics as stats

import numpy as np
import torch

from dynamic_maze_start_goal import (
    MazeTopology, load_trained_run, full_coverage_eval,
    DENSE, RSAE,
)

"""
Comparacao pareada entre o baseline denso (TorchDDQN) e o ReservedSAECollab
no ambiente goal-conditioned.

O ponto do script nao e "rodar dois treinos" — isso um loop de shell faz. O que
ele faz e garantir que a comparacao SEJA comparacao:

  1. Todo hiperparametro que nao e a arquitetura entra IDENTICO nos dois bracos.
     Os defaults por modelo em dynamic_maze_start_goal diferem em encoder,
     gamma, lr, batch, learn_interval, eps_decay_frac e min_replay; deixar
     qualquer um deles no default torna o resultado ininterpretavel.

  2. Depois dos treinos, le o bloco `effective` de cada summary.json e VERIFICA
     que o pareamento de fato aconteceu. Se algum campo divergir, aborta o
     relatorio. E a unica defesa contra "achei que tinha pareado".

  3. Avalia em COBERTURA TOTAL (todos os pares validos), nao no held-out de 256.
     Ja foi medido que o held-out pode inverter a ordem de dois checkpoints.

  4. Agrega varias seeds com media +- desvio. Um run so nao sustenta conclusao:
     o SAE oscilou 10 pontos entre a melhor e a ultima avaliacao do mesmo run.

USO
    python dynamic_compare.py --dry-run
    python dynamic_compare.py --seeds 333 334 335
    python dynamic_compare.py --report-only -o dynamic_results/cmp
    python dynamic_compare.py -h
"""


# Campos do summary.json que TEM de ser iguais entre os bracos. Se um destes
# divergir, a diferenca de desempenho nao pode ser atribuida a arquitetura.
PAIRED_TOP = [
    "maze", "encoder", "possible_actions", "num_last_states", "num_last_actions",
    "state_size", "episodes", "max_steps", "gamma", "use_her", "her_k",
    "n_step", "use_shaping", "curriculum", "eval_pairs", "seed",
]
PAIRED_EFFECTIVE = [
    "lr", "batch_size", "learn_interval", "eps_decay_frac", "min_replay",
    "her_k", "n_step",
]


# ─────────────────────────────────────────────────────────────────────────────
# Montagem dos comandos
# ─────────────────────────────────────────────────────────────────────────────

def shared_flags(a: argparse.Namespace) -> List[str]:
    """Flags identicas nos dois bracos. Tudo aqui e explicito de proposito:
    nada pode cair no default-por-modelo."""
    return [
        "--maze",           a.maze,
        "--encoder",        a.encoder,
        "--episodes",       str(a.episodes),
        "--gamma",          str(a.gamma),
        "--lr",             str(a.lr),
        "--batch",          str(a.batch),
        "--learn",          str(a.learn),
        "--eps-decay-frac", str(a.eps_decay_frac),
        "--min-replay",     str(a.min_replay),
        "--n-step",         str(a.n_step),
        "--her-k",          str(a.her_k),
        "--eval-pairs",     str(a.eval_pairs),
        "--eval-interval",  str(a.eval_interval),
    ] + (["--possible-actions"] if a.possible_actions
         else ["--no-possible-actions"])


def arm_flags(arm: str, a: argparse.Namespace) -> List[str]:
    """O que difere: so a arquitetura."""
    if arm == DENSE:
        return ["--hidden"] + [str(h) for h in a.dense_hidden]
    return [
        "--sae-hidden",      str(a.sae_hidden),
        "--sae-extra",       str(a.sae_extra),
        "--max-layers",      str(a.max_layers),
        "--insertion",       a.insertion,
        "--layer-mode",      a.layer_mode,
        "--mutation",        a.mutation,
        "--new-layer-lr",    str(a.new_layer_lr),
        "--insert-criterion", a.insert_criterion,
        "--insert-patience-evals", str(a.insert_patience_evals),
        "--insert-min-delta",      str(a.insert_min_delta),
        "--insert-warmup-frac",    str(a.insert_warmup_frac),
        "--insert-tail-frac",      str(a.insert_tail_frac),
        "--insert-regress-guard",  str(a.insert_regress_guard),
        "--insert-skip-success",   str(a.insert_skip_success),
    ]


def run_dir_for(a: argparse.Namespace, arm: str, seed: int) -> str:
    return os.path.join(a.out, arm, str(seed))


def build_cmd(a: argparse.Namespace, arm: str, seed: int) -> List[str]:
    return (
        [sys.executable, "dynamic_maze_start_goal.py", "--model", arm,
         "--seed", str(seed), "--dir", run_dir_for(a, arm, seed)]
        + shared_flags(a) + arm_flags(arm, a)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Execucao
# ─────────────────────────────────────────────────────────────────────────────

def run_training(cmd: List[str], tag: str, log_path: str) -> Tuple[int, float]:
    """Roda um treino, ecoando a saida prefixada e salvando o log completo."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.perf_counter()

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=env,
        )
        for line in proc.stdout:
            log.write(line)
            line = line.rstrip()
            # so ecoa o que interessa; o log guarda tudo
            if line.startswith("[ep") or line.startswith("[AVISO") \
               or line.startswith("  FINAL"):
                print(f"  {tag} | {line}")
        proc.wait()

    return proc.returncode, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Verificacao do pareamento
# ─────────────────────────────────────────────────────────────────────────────

def check_pairing(summaries: Dict[Tuple[str, int], dict]) -> List[str]:
    """Confere que os campos pareados sao iguais entre os bracos, seed a seed.

    Retorna a lista de divergencias. Vazia = comparacao valida.
    """
    problems: List[str] = []
    seeds = sorted({seed for (_, seed) in summaries})

    for seed in seeds:
        a = summaries.get((DENSE, seed))
        b = summaries.get((RSAE, seed))
        if a is None or b is None:
            continue
        for f in PAIRED_TOP:
            if f == "seed":
                continue
            va, vb = a.get(f), b.get(f)
            if va != vb:
                problems.append(f"seed {seed}: {f} difere -> "
                                f"{DENSE}={va!r}  {RSAE}={vb!r}")
        ea, eb = a.get("effective", {}), b.get("effective", {})
        for f in PAIRED_EFFECTIVE:
            va, vb = ea.get(f), eb.get(f)
            if va != vb:
                problems.append(f"seed {seed}: effective.{f} difere -> "
                                f"{DENSE}={va!r}  {RSAE}={vb!r}")
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Relatorio
# ─────────────────────────────────────────────────────────────────────────────

def mean_sd(xs: Sequence[float]) -> Tuple[float, float]:
    xs = [x for x in xs if x == x]                     # descarta nan
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.mean(xs), stats.stdev(xs)


def fmt(m: float, s: float, unit: str = "%") -> str:
    if m != m:
        return "     -   "
    return f"{m:5.1f}{unit} +-{s:4.1f}"


def print_report(results: List[dict], out_dir: str, ckpts: Sequence[str]) -> None:
    arms  = [DENSE, RSAE]
    seeds = sorted({r["seed"] for r in results})

    print()
    print("=" * 78)
    print("  COBERTURA TOTAL (todos os pares validos, politica deterministica)")
    print("=" * 78)

    for ckpt in ckpts:
        print(f"\n  -- {ckpt} " + "-" * (66 - len(ckpt)))
        header = f"  {'braco':<14}" + "".join(f"{('s'+str(s)):>10}" for s in seeds) \
                 + f"{'media':>16}{'params':>12}"
        print(header)
        for arm in arms:
            rows = [r for r in results if r["arm"] == arm and r["ckpt"] == ckpt]
            if not rows:
                continue
            by_seed = {r["seed"]: r for r in rows}
            cells = "".join(
                f"{by_seed[s]['success_rate']:9.1f}%" if s in by_seed else f"{'-':>10}"
                for s in seeds
            )
            m, sd  = mean_sd([r["success_rate"] for r in rows])
            params = int(stats.mean([r["parameters"] for r in rows]))
            print(f"  {arm:<14}{cells}  {fmt(m, sd)}{params:>12,}")

        d = [r["success_rate"] for r in results
             if r["arm"] == DENSE and r["ckpt"] == ckpt]
        s_ = [r["success_rate"] for r in results
              if r["arm"] == RSAE and r["ckpt"] == ckpt]
        if d and s_:
            md, sdd = mean_sd(d)
            ms, sds = mean_sd(s_)
            delta   = ms - md
            # Criterio deliberadamente conservador: com poucas seeds, qualquer
            # coisa dentro de 1 desvio combinado nao sustenta conclusao.
            pooled  = (sdd ** 2 + sds ** 2) ** 0.5
            verdict = ("INCONCLUSIVO (dentro do ruido entre seeds)"
                       if abs(delta) <= pooled or len(d) < 2
                       else f"{RSAE if delta > 0 else DENSE} melhor")
            print(f"\n  delta ({RSAE} - {DENSE}): {delta:+.1f}pp   "
                  f"desvio combinado {pooled:.1f}pp   -> {verdict}")
            if len(d) < 3:
                print(f"  [AVISO] so {len(d)} seed(s) por braco. "
                      f"Use --seeds com 3+ para que a media signifique algo.")

    # ── alcance por distancia, no ultimo checkpoint listado ─────────────────
    ckpt = ckpts[-1]
    print(f"\n  -- alcance por distancia otima ({ckpt}, media entre seeds) " + "-" * 8)
    dists = sorted({d for r in results if r["ckpt"] == ckpt
                    for d in r["by_distance"]})
    print(f"  {'d':>3} {'pares':>7}" + "".join(f"{a:>16}" for a in arms))
    for dd in dists:
        line = f"  {dd:>3}"
        npair = 0
        for arm in arms:
            rows = [r for r in results if r["arm"] == arm and r["ckpt"] == ckpt]
            vals = []
            for r in rows:
                if str(dd) in r["by_distance"] or dd in r["by_distance"]:
                    ok, tot = r["by_distance"].get(dd) or r["by_distance"][str(dd)]
                    vals.append(100.0 * ok / tot)
                    npair = tot
            m, sd = mean_sd(vals)
            line += f"{fmt(m, sd):>16}" if m == m else f"{'-':>16}"
        print(f"  {dd:>3} {npair:>7}" + line[5:])

    print("=" * 78)
    print(f"  resultados -> {os.path.join(out_dir, 'comparison.json')}")
    print(f"                {os.path.join(out_dir, 'comparison.csv')}")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dynamic_compare.py",
        description="Comparacao pareada dense (TorchDDQN) vs ReservedSAECollab.",
        epilog="Os defaults abaixo pareiam o run SAE de referencia "
               "(one_hot, gamma 0.999, lr 1e-5, batch 512, learn 4, "
               "eps-decay-frac 1.0). --dense-hidden 1200 1500 1800 2100 da "
               "8,590,204 params contra 8,414,156 do SAE (+2.1%).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-o", "--out", default="dynamic_results/cmp")
    p.add_argument("--seeds", type=int, nargs="+", default=[333, 334, 335])
    p.add_argument("--arms", nargs="+", choices=[DENSE, RSAE],
                   default=[DENSE, RSAE])
    p.add_argument("--dry-run", action="store_true",
                   help="imprime os comandos e sai")
    p.add_argument("--report-only", action="store_true",
                   help="pula os treinos e so avalia o que ja existe em --out")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="nao retreina se summary.json ja existe (default)")
    p.add_argument("--force", dest="skip_existing", action="store_false",
                   help="retreina mesmo se ja existir")
    p.add_argument("--checkpoints", nargs="+",
                   default=["model_best.pth", "model_last.pth"])
    p.add_argument("--eval-max-pairs", type=int, default=None, metavar="N",
                   help="teto de pares na cobertura total (mazes grandes)")
    p.add_argument("--eval-device", default=None, choices=["cpu", "cuda"])

    s = p.add_argument_group("hiperparametros PAREADOS (iguais nos dois bracos)")
    s.add_argument("--maze", default="./mazes/small_eg.maze")
    s.add_argument("--encoder", default="one_hot")
    s.add_argument("--possible-actions", dest="possible_actions",
                   action="store_true", default=True)
    s.add_argument("--no-possible-actions", dest="possible_actions",
                   action="store_false")
    s.add_argument("--episodes", type=int,   default=3000)
    s.add_argument("--gamma",    type=float, default=0.999)
    s.add_argument("--lr",       type=float, default=1e-5)
    s.add_argument("--batch",    type=int,   default=512)
    s.add_argument("--learn",    type=int,   default=4)
    s.add_argument("--eps-decay-frac", type=float, default=1.0)
    s.add_argument("--min-replay",     type=int,   default=1024)
    s.add_argument("--n-step",   type=int, default=3)
    s.add_argument("--her-k",    type=int, default=4)
    s.add_argument("--eval-pairs",    type=int, default=256)
    s.add_argument("--eval-interval", type=int, default=50)

    d = p.add_argument_group("braco denso")
    d.add_argument("--dense-hidden", type=int, nargs="+",
                   default=[1200, 1500, 1800, 2100], metavar="N")

    r = p.add_argument_group("braco ReservedSAECollab")
    r.add_argument("--sae-hidden", type=int,   default=1024)
    r.add_argument("--sae-extra",  type=int,   default=256)
    r.add_argument("--max-layers", type=int,   default=4)
    r.add_argument("--insertion",  default="CRT")
    r.add_argument("--layer-mode", default="M4")
    r.add_argument("--mutation",   default="Hidden")
    r.add_argument("--new-layer-lr", type=float, default=5e-5)
    r.add_argument("--insert-criterion", default="plateau",
                   choices=["plateau", "slope", "variance"])
    r.add_argument("--insert-patience-evals", type=int,   default=2,
                   help="janela de avaliacoes do criterio (default: %(default)s)")
    r.add_argument("--insert-min-delta",      type=float, default=1.0,
                   help="pp de ganho que contam como progresso (default: %(default)s)")
    r.add_argument("--insert-warmup-frac",    type=float, default=0.25,
                   help="[slope] fracao inicial sem insercao (default: %(default)s)")
    r.add_argument("--insert-tail-frac",      type=float, default=0.20,
                   help="[slope] fracao final sem insercao (default: %(default)s)")
    r.add_argument("--insert-regress-guard",  type=float, default=5.0,
                   help="[slope] pp abaixo do melhor que bloqueiam a insercao "
                        "(default: %(default)s)")
    r.add_argument("--insert-skip-success",    type=float, default=95.0,
                   help="acima desta taxa no held-out nenhuma camada e ativada "
                        "(default: %(default)s)")
    return p


def main() -> int:
    a = build_parser().parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = [(arm, seed) for seed in a.seeds for arm in a.arms]

    # ── dry run ─────────────────────────────────────────────────────────────
    if a.dry_run:
        for arm, seed in jobs:
            print(f"\n# {arm} seed {seed}")
            print("  " + " ".join(build_cmd(a, arm, seed)))
        print()
        return 0

    # ── treinos ─────────────────────────────────────────────────────────────
    if not a.report_only:
        print("=" * 78)
        print(f"  {len(jobs)} treinos: {a.arms} x seeds {a.seeds}")
        print(f"  saida: {a.out}")
        print("=" * 78)

        for i, (arm, seed) in enumerate(jobs, 1):
            rd  = run_dir_for(a, arm, seed)
            tag = f"{arm[:5]}/s{seed}"
            if a.skip_existing and os.path.exists(os.path.join(rd, "summary.json")):
                print(f"[{i}/{len(jobs)}] {tag}: ja existe, pulando "
                      f"(use --force para retreinar)")
                continue
            print(f"[{i}/{len(jobs)}] {tag}: treinando...")
            code, secs = run_training(
                build_cmd(a, arm, seed), tag,
                os.path.join(a.out, "logs", f"{arm}_{seed}.log"),
            )
            status = "OK" if code == 0 else f"FALHOU (exit {code})"
            print(f"[{i}/{len(jobs)}] {tag}: {status} em {secs/60:.1f} min")
            if code != 0:
                print(f"          log: {os.path.join(a.out,'logs',f'{arm}_{seed}.log')}")

    # ── coleta dos summaries e verificacao do pareamento ────────────────────
    summaries: Dict[Tuple[str, int], dict] = {}
    for arm, seed in jobs:
        sp = os.path.join(run_dir_for(a, arm, seed), "summary.json")
        if os.path.exists(sp):
            with open(sp) as f:
                summaries[(arm, seed)] = json.load(f)

    if not summaries:
        print("[ERROR] nenhum summary.json encontrado. Rode sem --report-only.")
        return 1

    problems = check_pairing(summaries)
    print()
    print("=" * 78)
    print("  VERIFICACAO DO PAREAMENTO")
    print("=" * 78)
    if problems:
        for p_ in problems:
            print(f"  [X] {p_}")
        print()
        print("  A comparacao NAO e valida: os bracos diferem em algo que nao e")
        print("  a arquitetura. Corrija as flags e rode com --force.")
        return 1
    print(f"  [OK] {len(PAIRED_TOP)-1} campos + {len(PAIRED_EFFECTIVE)} efetivos "
          f"identicos entre os bracos, em todas as seeds")

    # ── cobertura total ─────────────────────────────────────────────────────
    device = a.eval_device or ("cuda" if torch.cuda.is_available() else "cpu")
    results: List[dict] = []

    print()
    print("=" * 78)
    print(f"  AVALIANDO COBERTURA TOTAL  (device={device})")
    print("=" * 78)

    for (arm, seed), summ in sorted(summaries.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rd = run_dir_for(a, arm, seed)
        for ckpt in a.checkpoints:
            if not os.path.exists(os.path.join(rd, ckpt)):
                continue
            try:
                _, topo, env, agent, s, _ = load_trained_run(
                    rd, ckpt, device=device, verbose=False
                )
            except Exception as e:
                print(f"  [X] {arm}/s{seed}/{ckpt}: {e}")
                continue

            ev = full_coverage_eval(agent, env, topo, s["max_steps"],
                                    max_pairs=a.eval_max_pairs, seed=seed)
            results.append({
                "arm": arm, "seed": seed, "ckpt": ckpt,
                "success_rate": ev["success_rate"],
                "opt_ratio":    ev["opt_ratio"],
                "n_pairs":      ev["n_pairs"],
                "sampled":      ev["sampled"],
                "by_distance":  ev["by_distance"],
                "parameters":   s.get("parameters",
                                      sum(p.numel() for p in agent.policy_net.parameters())),
                "held_out_best": s.get("best_eval_success"),
                "held_out_final": s.get("final_success"),
            })
            print(f"  {arm:<14} s{seed} {ckpt:<16} "
                  f"{ev['success_rate']:5.1f}%  ({ev['n_pairs']:,} pares, "
                  f"opt_ratio {ev['opt_ratio']:.2f})")
            del agent, env, topo
            if device == "cuda":
                torch.cuda.empty_cache()

    if not results:
        print("[ERROR] nenhum checkpoint avaliado.")
        return 1

    with open(os.path.join(a.out, "comparison.json"), "w") as f:
        json.dump({"config": vars(a), "results": results}, f, indent=2)

    with open(os.path.join(a.out, "comparison.csv"), "w", newline="") as f:
        cols = ["arm", "seed", "ckpt", "success_rate", "opt_ratio", "n_pairs",
                "parameters", "held_out_best", "held_out_final"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print_report(results, a.out, a.checkpoints)
    return 0


if __name__ == "__main__":
    sys.exit(main())
