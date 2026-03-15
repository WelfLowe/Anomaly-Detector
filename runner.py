import argparse
import json
import os
from datetime import datetime
from TestAnomalyDetectors import run_anomaly_detectors, get_detectors


def get_experiment_dimensions(folder_path):
    dims = {'val': {'sets': 0, 'sevs': 0, 'runs': 0}, 'train': {'sets': 0, 'sevs': 0, 'runs': 0}}

    for filename in os.listdir(folder_path):
        if not filename.endswith('.npy'): continue

        parts = filename.replace('.npy', '').split('_')
        if len(parts) < 4: continue

        try:
            offset = 1 if filename.endswith('_labels') else 0
            s, sev, r = int(parts[-4 + offset]), int(parts[-3 + offset]), int(parts[-2 + offset])
            key = parts[0]

            if key in dims:
                dims[key]['sets'] = max(dims[key]['sets'], s)
                dims[key]['sevs'] = max(dims[key]['sevs'], sev)
                dims[key]['runs'] = max(dims[key]['runs'], r)
        except ValueError:
            continue

    if all(dims['val'][k] == dims['train'][k] for k in ['sets', 'sevs', 'runs']):
        t = dims['train']
        return t['sets'] + 1, t['sevs'] + 1, t['runs'] + 1

    return -1, -1, -1


def generate_manifest(tag):
    _, n_sevs, n_runs = get_experiment_dimensions('testsets/')
    if n_runs == -1:
        print("Mismatch between train and validation test sets.")
        return False

    algo_info = []
    for d in get_detectors():
        info = {"name": d.get_name()}
        # NOTE: If your detector classes have a method for hyperparams, fetch it here
        if hasattr(d, 'get_params'): 
            info["params"] = d.get_params()
        algo_info.append(info)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "n_severities": n_sevs,
        "n_runs": n_runs,
        "tag": tag,
        "algorithms": algo_info
    }
    
    with open(f"res_files/{tag}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=0)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args()

    if args.setup:
        exit(0 if generate_manifest(args.tag) else 1)

    _, n_sevs, n_runs = get_experiment_dimensions('testsets/')
    run_anomaly_detectors(args.dataset, n_sevs, n_runs, args.tag)