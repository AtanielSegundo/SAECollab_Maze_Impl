import os
import numpy as np

from typing import *

def get_specific_directories(root_dir, target_name):
    specific_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if target_name in dirname:
                specific_dirs.append(os.path.join(dirpath, dirname))
    return specific_dirs

class SearchFilter:
    def __init__(self,*names):
        self.allowed = names

def traversal_search(base: str,
                     targets: Dict[str, str],
                     filters: List[SearchFilter]) -> List[Dict[str, List[str]]]:

    search_result: List[Dict[str, List[str]]] = [
        {key: [] for key in targets.keys()} for _ in filters
    ]

    for dirpath, dirnames, filenames in os.walk(base):

        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)

            for f_idx, f in enumerate(filters):

                if not any(name in fullpath for name in f.allowed):
                    continue

                for target_key, target_tail in targets.items():
                    if fullpath.replace("\\", "/").endswith(target_tail):
                        search_result[f_idx][target_key].append(fullpath)

    return search_result


def main(*args,**kwargs):
    search_base = "./test/333"

    search_targets = {
        "baseline": "dense_model/metrics.csv",
        "sae"     : "sae_tolerance_model/metrics.csv"
    }

    search_filters = [
        SearchFilter("M1"),
        SearchFilter("M2"),
        SearchFilter("M3"),
        SearchFilter("M4")
    ]

    search_result = traversal_search(search_base,search_targets,search_filters)

    [print(p) for p in search_result[0]["baseline"]]
    [print(p) for p in search_result[0]["sae"]]

    print(len(search_result[0]["baseline"]))
    print(len(search_result[0]["sae"]))

    """
        SEARCH RESULT SHOULD SUPPORT SOMETHING LIKE:

            baseline_result:List[str] = search_result[0].get("baseline",None)
                                        WHERE search_result[0] is related to SearchFilter("M1")  
    """

if __name__ == "__main__":
    main()
    