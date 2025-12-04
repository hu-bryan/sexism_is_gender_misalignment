"""
Entry point for the gender misalignment experiment.

This script:

1. Calls gender_misalignment_experiment.experiment.run_experiment().
2. Uses the returned objects (directions, fit results, eval stats).
3. Saves them into a run-specific folder under ./experiment_runs/
   for later postprocessing (tables, plots, etc).
"""

import json
from datetime import datetime
from pathlib import Path

import torch

from sexism_experiment.experiment import run_experiment
from sexism_experiment.config import BASE_OUTPUT_DIR


def save_direction_dict(direction, path):
    """
    Save a direction dict {layer: tensor} as a .pt file plus norms JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cpu_dict = {str(k): v.cpu() for k, v in direction.items()}
    torch.save(cpu_dict, str(path))

    norms = {str(k): float(v.norm().item()) for k, v in direction.items()}
    with open(path.with_suffix(path.suffix + ".norms.json"), "w") as f:
        json.dump(norms, f, indent=2)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    # 1. Run the core experiment (this will ALSO save things under BASE_OUTPUT_DIR)
    results = run_experiment()

    # 2. Create a run-specific folder for postprocessing outputs
    #    e.g. ./experiment_runs/2025-12-03_14-25-10/
    runs_root = Path("experiment_runs")
    runs_root.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Saving run outputs to:", run_dir)

    gender_dir = results["gender_dir"]
    general_dir = results["general_dir"]
    sexism_dir = results["sexism_dir"]
    fit_results = results["fit_results"]
    baseline_eval = results["baseline_eval"]
    steered_eval = results["steered_eval"]

    # 3. Save direction dicts into this run-specific folder
    save_direction_dict(gender_dir, run_dir / "gender_semantic_direction.pt")
    save_direction_dict(general_dir, run_dir / "general_misalignment_direction.pt")
    save_direction_dict(sexism_dir, run_dir / "sexism_direction.pt")

    # 4. Save JSON summaries (fit + evaluation stats)
    save_json(fit_results, run_dir / "sexism_linear_fit.json")
    save_json(baseline_eval, run_dir / "gender_sexism_eval_baseline.json")
    save_json(steered_eval, run_dir / "gender_sexism_eval_steered.json")

    # 5. Optionally record where BASE_OUTPUT_DIR was, for reference
    meta = {
        "base_output_dir": str(BASE_OUTPUT_DIR),
    }
    save_json(meta, run_dir / "meta.json")

    print("Done. You can now postprocess everything under:", run_dir)


if __name__ == "__main__":
    main()
