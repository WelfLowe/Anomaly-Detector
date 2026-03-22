# Anomaly detection in unlabeled signals

## General Description
This is a repository for the paper: *Anomaly detection in unlabeled signals*. Instructions on how to repreoduce the main experimental findings of the paper are given below.

## How to Test Run

To execute the full pipeline, from data generation to anomaly detection and visualization, ensure your environment is set up and follow the steps below.

### Setup Environment
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Create Data
Generate the synthetic time-series datasets (train and validation pairs) featuring various anomaly types (Lift, Noise, Shift, Amplitude, Blackout).

```bash
python TestsetGeneration.py
```
*Outputs are saved to `testsets/set1/` along with a `manifest.json`.*

### Step 2: Run Anomaly Detectors
Execute the anomaly detection algorithms across the generated datasets. The bash script sets up the experiment dimensions and parallelizes the runs using `tmux`.

```bash
# Optional: Provide a custom tag, otherwise a timestamp is used
./run_script.sh my_custom_experiment
```
*Note: Because the generation script outputs to `testsets/set1`, ensure your `run_script.sh` or `runner.py` is pointing to this directory (e.g., updating the default `--testset_dir` flag).*

Results and the execution manifest are saved to `res_files/<TAG>/`. To monitor the background tasks, attach to the tmux sessions (e.g., `tmux attach -t run_<TAG>_0`).

### Step 3: Plot Results
In order to create visualization of the results its possible to use our `comparison_ploy.py` located inseide the `viz` folder.

```bash
# Example
python viz/comparison_plot.py
```
*Note: You need to change the tag inside the `viz/comparison_plot.py`-file.*

## Citation
If you utilize this package or codebase in your work, please cite the following paper:

```bibtex
@article{Viberg2026anomaly,
  title={Anomaly detection in unlabeled signals},
  author={Viberg, Felix and Nordqvist, Jonas and L\"{o}we, Welf},
  journal={None},
  year={2026}
}