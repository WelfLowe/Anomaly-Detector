import argparse
import json
import os
import re
from datetime import datetime
from TestAnomalyDetectors import run_anomaly_detectors, get_detectors

def get_experiment_dimensions(folder_path):
    dims = {'val': {'sevs': 0, 'runs': 0}, 'train': {'sevs': 0, 'runs': 0}}
    # TODO: Expand regex if dynamic testset IDs are added
    pattern = re.compile(r"^(train|val)_.+_(\d+)_(\d+)(_labels)?\.npy$")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if not match: 
            continue

        prefix, sev, r, _ = match.groups()
        dims[prefix]['sevs'] = max(dims[prefix]['sevs'], int(sev))
        dims[prefix]['runs'] = max(dims[prefix]['runs'], int(r))

    if all(dims['val'][k] == dims['train'][k] for k in ['sevs', 'runs']):
        return dims['train']['sevs'] + 1, dims['train']['runs'] + 1

    return -1, -1

def generate_manifest(tag, testset_dir):
    n_sevs, n_runs = get_experiment_dimensions(testset_dir)
    if n_runs == -1:
        print("Mismatch between train and validation test sets.")
        return False

    algo_info = []
    for d in get_detectors():
        info = {"name": d.get_name()}
        # NOTE: Fetching hyperparams if available
        if hasattr(d, 'get_params'): 
            info["params"] = d.get_params()
        algo_info.append(info)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "testset_dir": testset_dir,
        "n_severities": n_sevs,
        "n_runs": n_runs,
        "tag": tag,
        "algorithms": algo_info
    }
    
    os.makedirs(f"res_files/{tag}", exist_ok=True)
    with open(f"res_files/{tag}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="0")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--testset_dir", type=str, default="testsets/testsets_original")
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args()

    if args.setup:
        exit(0 if generate_manifest(args.tag, args.testset_dir) else 1)

    n_sevs, n_runs = get_experiment_dimensions(args.testset_dir)
    run_anomaly_detectors(args.dataset, n_sevs, n_runs, args.tag, args.testset_dir)