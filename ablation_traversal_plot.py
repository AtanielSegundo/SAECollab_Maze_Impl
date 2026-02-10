import os
from typing import *

import numpy as np
import math
from matplotlib import pyplot as plt

from Ablation import ModelTrainMetrics

def get_specific_directories(root_dir, target_name):
    specific_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if target_name in dirname:
                specific_dirs.append(os.path.join(dirpath, dirname))
    return specific_dirs

class SearchFilter:
    def __init__(self, *names):
        self.allowed = names

SearchResult: TypeAlias = Dict[str, Dict[str, List[str]]]

def traversal_search(base: str,
                     targets: Dict[str, str],
                     filters: Dict[str, SearchFilter]
) -> SearchResult:
    search_result: SearchResult = {
        f_key: {key: [] for key in targets.keys()} 
        for f_key in filters.keys()
    }
        
    for dirpath, _, filenames in os.walk(base):
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)

            for f_key, f in filters.items():
                if not any(name in fullpath for name in f.allowed):
                    continue

                for target_key, target_tail in targets.items():
                    if fullpath.replace("\\", "/").endswith(target_tail):
                        search_result[f_key][target_key].append(fullpath)

    return search_result

class CompiledTrainMetrics:
    def __init__(self, metrics_keys: list[str]):
        # The Key Array should be like: shape:= [N,2] where 2 = (mean,variance)
        self.comp_metrics = {key: [] for key in metrics_keys}
        self.comp_metrics_count = {key: [] for key in metrics_keys}
    
    def compile_metrics(self, m: ModelTrainMetrics):
        for key, compile_arr in self.comp_metrics.items():
            maped_key = m.header_map.get(key,None)
            if maped_key in m.available_metrics:
                arr = m.__dict__[key]
                for idx, elem in enumerate(arr):
                    if idx >= len(compile_arr):
                        # The mean should be the element itself and the variance is 0
                        compile_arr.append((elem, 0))
                        self.comp_metrics_count[key].append(1)
                    else:
                        # ONLINE MEAN AND ONLINE VARIANCE UPDATE
                        old_mean, old_var = compile_arr[idx]
                        old_count = self.comp_metrics_count[key][idx]

                        new_count = old_count + 1

                        # Update mean: mean_new = mean_old + (x - mean_old) / n
                        delta = elem - old_mean
                        new_mean = old_mean + delta / new_count

                        # Update variance (Welford's method)
                        # M2 = sum of squared differences from mean
                        old_M2 = old_var * old_count
                        delta2 = elem - new_mean
                        new_M2 = old_M2 + delta * delta2
                        new_var = new_M2 / new_count

                        # Update stored values
                        compile_arr[idx] = (new_mean, new_var)
                        self.comp_metrics_count[key][idx] = new_count
    
    def extend(self, other: "CompiledTrainMetrics"):
        """
        Merge another CompiledTrainMetrics into this one,
        combining means/variances/counts correctly.
        """
        for key in self.comp_metrics.keys():
            arr_self = self.comp_metrics[key]
            cnt_self = self.comp_metrics_count[key]

            arr_other = other.comp_metrics.get(key, [])
            cnt_other = other.comp_metrics_count.get(key, [])

            for idx, (mean_o, var_o) in enumerate(arr_other):
                count_o = cnt_other[idx]

                if idx >= len(arr_self):
                    # just copy
                    arr_self.append((mean_o, var_o))
                    cnt_self.append(count_o)
                else:
                    mean_s, var_s = arr_self[idx]
                    count_s = cnt_self[idx]

                    # combine two distributions
                    new_count = count_s + count_o
                    if new_count == 0:
                        continue

                    delta = mean_o - mean_s
                    new_mean = mean_s + delta * (count_o / new_count)

                    M2_s = var_s * count_s
                    M2_o = var_o * count_o
                    new_M2 = M2_s + M2_o + delta * delta * (count_s * count_o / new_count)
                    new_var = new_M2 / new_count

                    arr_self[idx] = (new_mean, new_var)
                    cnt_self[idx] = new_count


def create_compiled_metrics_structure(filters: Dict[str, SearchFilter],
                                       targets: Dict[str, str],
                                       metrics_keys: List[str]) -> Dict[str, Dict[str, CompiledTrainMetrics]]:
    """
    Create nested dictionary structure for compiled metrics.
    
    Returns:
        {filter_key: {target_key: CompiledTrainMetrics}}
    """
    return {
        f_key: {
            t_key: CompiledTrainMetrics(metrics_keys)
            for t_key in targets.keys()
        }
        for f_key in filters.keys()
    }


def compile_all_metrics(search_result: SearchResult,
                        compiled_metrics: Dict[str, Dict[str, CompiledTrainMetrics]]) -> None:
    """
    Load and compile all metrics from search results into compiled metrics structure.
    
    Args:
        search_result: Nested dict of {filter: {target: [paths]}}
        compiled_metrics: Nested dict of {filter: {target: CompiledTrainMetrics}}
    """
    for f_key, comp_metrics_dict in compiled_metrics.items():
        targets_metrics_paths = search_result[f_key]
        
        for t_key, comp_metric in comp_metrics_dict.items():
            metrics_paths = targets_metrics_paths[t_key]
            
            for path in metrics_paths:
                metrics = ModelTrainMetrics.load(path)
                comp_metric.compile_metrics(metrics)


def print_search_summary(search_result: SearchResult) -> None:
    """Print summary of found metrics files."""
    print("=" * 60)
    print("SEARCH SUMMARY")
    print("=" * 60)
    for f_key, targets in search_result.items():
        print(f"\n{f_key}:")
        for t_key, paths in targets.items():
            print(f"  {t_key:15s}: {len(paths):4d} files")
    print("=" * 60)


def _extract_mean_std(comp, metric_key):
    """Retorna (means, stds) como numpy arrays. Se não existir, retorna ([],[])."""
    arr = comp.comp_metrics.get(metric_key, [])
    if not arr:
        return np.array([]), np.array([])
    means = np.array([t[0] for t in arr], dtype=float)
    vars_ = np.array([t[1] for t in arr], dtype=float)
    stds = np.sqrt(np.maximum(vars_, 0.0))
    return means, stds

def _smooth(arr: np.ndarray, window: Optional[int]):
    if window is None or window <= 1 or arr.size == 0:
        return arr
    w = np.ones(window) / window
    return np.convolve(arr, w, mode='same')


def plot_compiled_metrics(
    compiled_metrics: Dict[str, Dict[str, object]],
    f_keys: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
    metrics_to_plot: Optional[Sequence[str]] = None,
    extra_metrics: Optional[Dict[str, object]] = None,
    episodes: Optional[int] = None,
    smooth_window: Optional[int] = None,
    figsize: tuple = (14, 8),
    out_file: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
    x_lim: Optional[Tuple[int, int]] = None,
    # new args
    disable_variance_shade: bool = False,
    max_no_mean: bool = False,
    min_no_mean: bool = False,
):
    """
    Plot utility for compiled_metrics.

    New args:
        disable_variance_shade: if True, do not draw std shading.
        max_no_mean: if True, plot mean+std as the line instead of mean.
        min_no_mean: if True, plot mean-std as the line instead of mean.

    Note: max_no_mean and min_no_mean cannot both be True.
    """
    if max_no_mean and min_no_mean:
        raise ValueError("Cannot set both max_no_mean and min_no_mean to True.")

    # defaults
    if f_keys is None:
        f_keys = list(compiled_metrics.keys())
    if metrics_to_plot is None:
        metrics_to_plot = [
            "reward", "success_rate", "cumulative_goals", "loss", "parameters", "steps",
            "branchs", "delta_time", "cumulative_time"
        ]

    # determine target if not given
    if target is None:
        sample = next(iter(compiled_metrics.values()), None)
        if sample and len(sample) == 1:
            target = next(iter(sample.keys()))
        else:
            raise ValueError("target not specified and cannot be inferred. Provide `target` argument.")

    # prepare subplots grid (flexible)
    n_metrics = len(metrics_to_plot)
    ncols = min(3, n_metrics)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    # find max episodes if not provided
    max_len = 0
    for f in f_keys:
        comp = compiled_metrics.get(f, {}).get(target)
        if comp:
            for mk in metrics_to_plot:
                # Skip cumulative_time as it's computed from delta_time
                if mk == "cumulative_time":
                    continue
                m, s = _extract_mean_std(comp, mk)
                max_len = max(max_len, m.size)
    if extra_metrics:
        for comp in extra_metrics.values():
            if hasattr(comp, "comp_metrics"):
                for mk in metrics_to_plot:
                    if mk == "cumulative_time":
                        continue
                    m, s = _extract_mean_std(comp, mk)
                    max_len = max(max_len, m.size)

    eps = int(episodes) if episodes is not None else max_len
    x = np.arange(1, eps + 1)

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]

        if x_lim is not None:
            ax.set_xlim(x_lim)

        plotted_any = False

        # plot each f_key
        for i, f in enumerate(f_keys):
            comp = compiled_metrics.get(f, {}).get(target)
            if comp is None:
                continue
            
            # Handle cumulative_time specially
            if metric == "cumulative_time":
                delta_means, delta_stds = _extract_mean_std(comp, "delta_time")
                if delta_means.size == 0:
                    continue
                
                # Compute cumulative sum of means
                means = np.cumsum(delta_means)
                # For cumulative variance: Var(Sum) = Sum(Var) for independent variables
                stds = np.sqrt(np.cumsum(delta_stds**2))
            else:
                means, stds = _extract_mean_std(comp, metric)
                if means.size == 0:
                    continue

            # optionally trim/pad to eps
            if eps and means.size >= eps:
                means = means[:eps]
                stds = stds[:eps]
            elif eps and means.size < eps:
                pad = eps - means.size
                means = np.concatenate([means, np.full(pad, np.nan)])
                stds = np.concatenate([stds, np.full(pad, np.nan)])

            if smooth_window:
                means = _smooth(means, smooth_window)
                stds = _smooth(stds, smooth_window)

            # compute what to plot as the "line" according to flags
            if max_no_mean:
                plot_line = means + stds
            elif min_no_mean:
                plot_line = means - stds
            else:
                plot_line = means

            col = color_cycle[i % len(color_cycle)] if color_cycle else None
            label = f"{f}"
            ax.plot(x, plot_line, label=label, linewidth=1.7, color=col)

            # shade unless disabled
            if not disable_variance_shade:
                lower = means - stds
                upper = means + stds
                ax.fill_between(x, lower, upper, alpha=0.15, color=col)

            plotted_any = True

        # plot extra overlays if provided
        if extra_metrics:
            for j, (label, comp) in enumerate(extra_metrics.items()):
                # comp may be ModelTrainMetrics-like (with lists) or CompiledTrainMetrics
                if hasattr(comp, "comp_metrics"):
                    if metric == "cumulative_time":
                        delta_means, delta_stds = _extract_mean_std(comp, "delta_time")
                        if delta_means.size > 0:
                            means = np.cumsum(delta_means)
                            stds = np.sqrt(np.cumsum(delta_stds**2))
                        else:
                            means = np.array([])
                            stds = np.array([])
                    else:
                        means, stds = _extract_mean_std(comp, metric)
                else:
                    attr_map = {
                        "reward": "reward",
                        "sucess_rate": "sucess_rate",
                        "cumulative_goals": "cumulative_goals",
                        "loss": "loss",
                        "parameters_cnt": "parameters",
                        "steps": "steps",
                        "branchs": "branchs",
                        "delta_time": "delta_time"
                    }
                    
                    if metric == "cumulative_time":
                        arr = getattr(comp, "delta_time", [])
                        if arr:
                            means = np.cumsum(np.array(arr, dtype=float))
                            stds = np.zeros_like(means)
                        else:
                            means = np.array([])
                            stds = np.array([])
                    else:
                        arr = getattr(comp, attr_map.get(metric, metric), [])
                        means = np.array(arr, dtype=float) if arr is not None else np.array([])
                        stds = np.zeros_like(means)

                if means.size == 0:
                    continue

                if eps and means.size >= eps:
                    means = means[:eps]
                    stds = stds[:eps]
                elif eps and means.size < eps:
                    pad = eps - means.size
                    means = np.concatenate([means, np.full(pad, np.nan)])
                    stds = np.concatenate([stds, np.full(pad, np.nan)])

                if smooth_window:
                    means = _smooth(means, smooth_window)
                    stds = _smooth(stds, smooth_window)

                # apply same mean-mode logic for overlays
                if max_no_mean:
                    plot_line = means + stds
                elif min_no_mean:
                    plot_line = means - stds
                else:
                    plot_line = means

                col = color_cycle[(len(f_keys) + j) % len(color_cycle)] if color_cycle else None
                ax.plot(x, plot_line, label=label, linestyle='--', linewidth=1.6, color=col)
                if not disable_variance_shade:
                    lower = means - stds
                    upper = means + stds
                    ax.fill_between(x, lower, upper, alpha=0.12, color=col)
                plotted_any = True

        ax.set_title(metric)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        ax.legend()
        # sensible y-labels for common metrics
        if metric in ("reward",):
            ax.set_ylabel("Cumulative Reward")
        elif metric in ("sucess_rate",):
            ax.set_ylabel("Success Rate (%)")
            ax.set_ylim(0, 105)
        elif metric in ("cumulative_goals", "cumulative_goals"):
            ax.set_ylabel("Cumulative Goals")
        elif metric in ("loss",):
            ax.set_ylabel("TD Loss")
        elif metric in ("parameters_cnt", "parameters", "parameters_cnt"):
            ax.set_ylabel("Num Parameters")
        elif metric in ("steps",):
            ax.set_ylabel("Steps per Episode")
        elif metric in ("branchs",):
            ax.set_ylabel("Number of Branches")
        elif metric in ("delta_time",):
            ax.set_ylabel("Time per Episode (s)")
        elif metric in ("cumulative_time",):
            ax.set_ylabel("Cumulative Time (s)")

        # if nothing plotted, show placeholder text
        if not plotted_any:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes, alpha=0.5)

    # hide unused axes
    for k in range(n_metrics, len(axes)):
        axes[k].axis('off')

    main_title = title or f"Compiled metrics - target={target}"
    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_file:
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved plot: {out_file}")
        return out_file

    if show:
        plt.show()
    return fig

def ablation_traverse(search_base, search_targets, search_filters, plot_save_path):
    search_result = traversal_search(search_base, search_targets, search_filters)

    target_metrics_keys = list(ModelTrainMetrics().available_metrics)
    compiled_metrics = create_compiled_metrics_structure(
        search_filters,
        search_targets,
        target_metrics_keys
    )

    compile_all_metrics(search_result, compiled_metrics)
    
    all_baselines_compiled = CompiledTrainMetrics(target_metrics_keys)
    for f_key,compiled_metrics_d in compiled_metrics.items():
        all_baselines_compiled.extend(compiled_metrics_d['baseline'])
        
    plot_compiled_metrics(
        compiled_metrics,
        f_keys=list(search_filters.keys()),
        target='sae',
        metrics_to_plot=None,
        smooth_window=10,
        extra_metrics={"baseline": all_baselines_compiled},
        out_file=plot_save_path,
        show=True,
        title=os.path.basename(plot_save_path).split(".")[0].upper().replace("_"," "),
        x_lim = (1,len(all_baselines_compiled.comp_metrics[target_metrics_keys[0]])-1)
    )

    return compiled_metrics, search_result

def main(*args, **kwargs):
    search_base = kwargs.get("base", None) or "./test/333"

    search_targets = {
        "baseline": "dense_model/metrics.csv",
        "sae": "sae_tolerance_model/metrics.csv"
    }

    search_filters = {
        "M1": SearchFilter("M1"),
        "M2": SearchFilter("M2"),
        "M3": SearchFilter("M3"),
        "M4": SearchFilter("M4")
    }
    out_file = 'plots/compared_methods.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Hidden": SearchFilter("Hidden"),
        "Normal": SearchFilter("Normal"),
        "Out"   : SearchFilter("Out"),
    }
    out_file = 'plots/compared_mutations.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "ALT": SearchFilter("ALT"),
        "CNT": SearchFilter("CNT"),
        "CRT": SearchFilter("CRT"),
        "DRT": SearchFilter("DRT"),
    }
    out_file = 'plots/compared_insertion_modes.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "M1": SearchFilter("ALT","M1"),
        "M2": SearchFilter("ALT","M2"),
        "M3": SearchFilter("ALT","M3"),
        "M4": SearchFilter("ALT","M4"),
    }
    out_file = 'plots/compared_alts_methods.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Arch1": SearchFilter("4_H0.5O1E0.5_HReLUOIdentityEReLU_HTOTET"),
        "Arch2": SearchFilter("4_H0.5O1E0.25_HReLUOIdentityEIdentity_HTOTET"),
        "Arch3": SearchFilter("4_H0.25O1E0.5_HReLUOIdentityEReLU_HFOTEF"),
        "Arch4": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF"),
    }
    out_file = 'plots/compared_archs.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Arch4-ALT": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF","ALT"),
        "Arch4-CNT": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF","CNT"),
        "Arch4-CRT": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF","CRT"),
        "Arch4-DRT": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF","DRT"),
    }
    out_file = 'plots/compared_archs_insertion.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Linear": SearchFilter("EIdentity"),
        "ReLU": SearchFilter("EReLU"),
    }
    out_file = 'plots/compared_extra_linear_relu.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "COORDS": SearchFilter("COORDS"),
        "ONE_HOT": SearchFilter("ONE_HOT"),
    }
    out_file = 'plots/compared_encodes.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Bias": SearchFilter("HTOTET"),
        "NoBias": SearchFilter("HFOTEF"),
    }
    out_file = 'plots/compared_bias.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "ALL": SearchFilter("metrics.csv"),
    }
    out_file = 'plots/compared_sae_baseline.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

if __name__ == "__main__":
    from sys import argv
    
    base_path = argv[-1] if len(argv) > 1 else "./test/333"
    main(base=base_path)