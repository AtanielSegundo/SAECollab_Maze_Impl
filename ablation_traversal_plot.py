from sys import argv
from ablation.traversal import *
from pathlib import Path

def main(*args, **kwargs):
    search_base = kwargs.get("base", None) 
    output_dir  = kwargs.get("out_dir", None)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    search_targets = {
        "baseline": "dense_model/metrics.csv",
        "sae": "sae_tolerance_model/metrics.csv"
    }

    if not os.path.exists(search_base):
        print(f"[ERROR] Invalid Path: {search_base}")
        return

    if os.path.isfile(search_base):
        search_filters = {
            "ALL": SearchFilter("metrics.csv"),
        }
        out_file = output_dir / 'test.png'
        ablation_traverse(os.path.dirname(search_base),search_targets,search_filters,out_file)
        return

    search_filters = {
        "Small" : SearchFilter("small_eg"),
        "Medium": SearchFilter("medium_eg"),
        "Big"   : SearchFilter("big_eg"),
    }
    out_file = output_dir / 'compared_maps.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "M1": SearchFilter("M1"),
        "M2": SearchFilter("M2"),
        "M3": SearchFilter("M3"),
        "M4": SearchFilter("M4")
    }
    out_file = output_dir / 'compared_methods.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Hidden": SearchFilter("Hidden"),
        "Normal": SearchFilter("Normal"),
        "Out"   : SearchFilter("Out"),
    }
    out_file = output_dir / 'compared_mutations.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "ALT": SearchFilter("ALT"),
        "CNT": SearchFilter("CNT"),
        "CRT": SearchFilter("CRT"),
        "DRT": SearchFilter("DRT"),
    }
    out_file = output_dir / 'compared_insertion_modes.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "M1": SearchFilter("ALT","M1"),
        "M2": SearchFilter("ALT","M2"),
        "M3": SearchFilter("ALT","M3"),
        "M4": SearchFilter("ALT","M4"),
    }
    out_file = output_dir / 'compared_alts_methods.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Arch1": SearchFilter("4_H0.5O1E0.5_HReLUOIdentityEReLU_HTOTET"),
        "Arch2": SearchFilter("4_H0.5O1E0.25_HReLUOIdentityEIdentity_HTOTET"),
        "Arch3": SearchFilter("4_H0.25O1E0.5_HReLUOIdentityEReLU_HFOTEF"),
        "Arch4": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF"),
    }
    out_file = output_dir / 'compared_archs.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Arch4-ALT": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF","ALT"),
        "Arch4-CNT": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF","CNT"),
        "Arch4-CRT": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF","CRT"),
        "Arch4-DRT": SearchFilter("4_H0.25O1E0.25_HReLUOIdentityEIdentity_HFOTEF","DRT"),
    }
    out_file = output_dir / 'compared_archs_insertion.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Linear": SearchFilter("EIdentity"),
        "ReLU": SearchFilter("EReLU"),
    }
    out_file = output_dir / 'compared_extra_linear_relu.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "COORDS": SearchFilter("COORDS","COORDS_NORM_nls_2_vcnt"),
        "ONE_HOT": SearchFilter("ONE_HOT","ONE_HOT_paf"),
    }
    out_file = output_dir / 'compared_encodes.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "Bias": SearchFilter("HTOTET"),
        "NoBias": SearchFilter("HFOTEF"),
    }
    out_file = output_dir / 'compared_bias.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

    search_filters = {
        "ALL": SearchFilter("metrics.csv"),
    }
    out_file = output_dir / 'compared_sae_baseline.png'
    ablation_traverse(search_base,search_targets,search_filters,out_file)

if __name__ == "__main__":

    expected_args = {
        "base": "./test/333",
        "out_dir": "./traversal_plots/"
    }
    
    _args = argv[1:]
    for idx,(key,val) in enumerate(expected_args.items()):
        if idx >= len(_args): 
            break
        expected_args[key] = _args[idx]

    main(**expected_args)