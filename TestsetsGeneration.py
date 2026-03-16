import json
import os
import numpy as np
import concurrent.futures
from tqdm import tqdm

from testset_generator.DatasetGeneratorBlackout import DatasetGeneratorBlackout
from testset_generator.DatasetGeneratorLift import DatasetGeneratorLift
from testset_generator.DatasetGeneratorSeason import (DatasetGeneratorNoise, DatasetGeneratorAmplitude, DatasetGeneratorShift)

def generate_dataset_pair(n, gen_idx, generator, s, K, N, base_dir):
    sev_label = int(s * 29 / 4)
    
    # NOTE: Using 'fraction' to match the original DatasetGenerator signature
    generator.generateKN(
        K, N, fraction=0.05, severeness=s, verbose=False,
        name=f"{base_dir}/train_{gen_idx}_{sev_label}_{n}"
    )
    generator.generateKN(
        K // 10, N, fraction=0.5, severeness=s, verbose=False,
        name=f"{base_dir}/val_{gen_idx}_{sev_label}_{n}"
    )

def write_manifest(output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(metadata, f, indent=4)

output_directory = "testsets_new_new"
K_train, N_points = 1000, 500
n_sets = 50
severities = np.linspace(0, 4, 30)

generators = [
    DatasetGeneratorLift(),
    DatasetGeneratorNoise(),
    DatasetGeneratorShift(),
    DatasetGeneratorAmplitude(),
    DatasetGeneratorBlackout()
]

manifest_data = {
    "K_train": K_train,
    "K_val": K_train // 10,
    "N_points": N_points,
    "train_fraction": 0.05,
    "val_fraction": 0.5,
    "n_sets": n_sets,
    "severities": severities.tolist(),
    "generators": [type(g).__name__ for g in generators]
}

write_manifest(output_directory, manifest_data)

with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = []
    for n in range(n_sets):
        for i, gen in enumerate(generators):
            for s in severities:
                futures.append(
                    executor.submit(generate_dataset_pair, n, i, gen, s, K_train, N_points, output_directory)
                )
    
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating Datasets"):
        # NOTE: Surfaces any hidden worker exceptions
        future.result()