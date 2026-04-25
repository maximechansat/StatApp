#!/usr/bin/env python3
"""
Batch execution script for running all YOLO scaling law experiments.
This script runs all 60 combinations automatically.
"""

import subprocess
import sys
import time


def run_experiment(seed, dataset_fraction, model_variant, resolution):
    """Run a single experiment"""
    cmd = [
        sys.executable, "scaling_law_study.py",
        "--seed", str(seed),
        "--dataset_fraction", str(dataset_fraction),
        "--model_variant", model_variant,
        "--resolution", str(resolution)
    ]
    
    print(f"Running: {seed=} | {model_variant} | {dataset_fraction*100:.0f}% | {resolution}px")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"   Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   Failed: {e}")
        print(f"   Error: {e.stderr}")
        return False

def main():
    # All combinations
    seeds = [4221376603, 3810243382, 693763239]
    dataset_fractions = [0.1, 0.25, 0.5, 1.0]
    model_variants = ["yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    resolutions = [416, 640, 1280]
    
    total_experiments = len(dataset_fractions) * len(model_variants) * len(resolutions) * len(seeds)
    completed = 0
    failed = 0
    
    print(f"Starting batch execution of {total_experiments} experiments")
    print("=" * 60)
    
    start_time = time.time()
    for model_variant in model_variants:
        for resolution in resolutions:
            for seed in seeds:
                for dataset_fraction in dataset_fractions:
                    completed += 1
                    print(f"\nExperiment {completed}/{total_experiments}")
                    success = run_experiment(seed, dataset_fraction, model_variant, resolution)
                    if not success:
                        failed += 1
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (total_experiments - completed) * avg_time
                    
                    print(f"   Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
                    print(f"   Elapsed: {elapsed/3600:.1f}h | Remaining: {remaining/3600:.1f}h")
    
    total_time = time.time() - start_time
    
    print(f"\nBatch execution completed!")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Successful: {total_experiments - failed}")
    print(f"   Failed: {failed}")
    print(f"   Total time: {total_time/3600:.1f} hours")
    
    if failed > 0:
        print(f"\n{failed} experiments failed. Check the output above for details.")
        sys.exit(1)
    else:
        print(f"\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
