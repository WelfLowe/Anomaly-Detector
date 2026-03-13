import argparse
from TestAnomalyDetectors import TestAnomalyDetector
import os


def find_args(folder_path):
    data = {
        'val': {
            'sets': 0,
            'sevs': 0,
            'runs': 0
        },
        'train': {
            'sets': 0,
            'sevs': 0,
            'runs': 0
        }
    }

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            parts = file_name.split('_')
            if len(parts) >= 4:
                try:
                    X, Y, Z = (int(parts[-4]), int(parts[-3]), int(
                        parts[-2])) if file_name.endswith('_labels.npy') else (
                            int(parts[-3]), int(parts[-2]),
                            int(parts[-1].replace('.npy', '')))
                    key = parts[0]
                    if key in data:
                        data[key]['sets'] = max(data[key]['sets'], X)
                        data[key]['sevs'] = max(data[key]['sevs'], Y)
                        data[key]['runs'] = max(data[key]['runs'], Z)
                except ValueError:
                    print(
                        f"Filen {file_name} innehåller icke-numeriska värden och ignoreras."
                    )
    if all(data['val'][k] == data['train'][k]
           for k in ['sets', 'sevs', 'runs']):
        return data['train']['sets'] + 1, data['train']['sevs'] + 1, data[
            'train']['runs'] + 1
    else:
        print(data['val'], data['train'])
        print("Train och validation skiljer sig i test-foldern")
        return -1, -1, -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument(
        "--dataset",
        dest="dset",
        default="0",
        help="Which dataset id to run on (0,1,...,4)",
    )
    args = parser.parse_args()
    _, n_severities, n_runs = find_args('testsets_new/')
    TestAnomalyDetector(args.dset, n_severities, n_runs)
