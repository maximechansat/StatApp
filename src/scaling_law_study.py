# YOLO Scaling Law Study - Individual Experiment Execution
# Run single experiments with specific parameters and configurable seeds

import json
import time
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Subset
from ultralytics import YOLO
from thop import profile
import random
import argparse
import yaml
import gc
import warnings
warnings.filterwarnings('ignore')

class ScalingLawStudy:
    """
    Individual experiment execution for YOLO scaling law study.
    
    Run single experiments with specific parameters:
    - dataset_fraction: 0.1, 0.25, 0.5, 1.0
    - model_variant: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    - resolution: 416, 640, 1280
    - seed: Integer for deterministic variance testing
    """
    
    def __init__(self, config_path="config.yaml", seed=42):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.root_dir = Path(self.config['dataset']['root_dir'])
        self.yaml_file = self.config['dataset']['yaml_file']
        self.results_dir = Path(self.config['results']['output_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.epochs = self.config['study']['epochs']
        self.seed = seed
        
        # Set seeds for reproducibility
        self._set_seeds()
        
        # Load existing results
        self._load_existing_results()
        
    def _set_seeds(self):
        """Set all random seeds for deterministic workflow"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)
        print(f"Random seed globally set to: {self.seed}")
        
    def _load_existing_results(self):
        """Load existing results"""
        results_file = self.results_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded {len(self.results)} existing results")
        else:
            self.results = []
            print("Starting fresh - no existing results found")
    
    def _save_results(self):
        """Save results to JSON (appends naturally as self.results grows)"""
        results_file = self.results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as CSV for easy analysis
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.results_dir / "results.csv"
            df.to_csv(csv_file, index=False)
        
        print(f"Results saved ({len(self.results)} total experiments in log)")
    
    def _create_subset_yaml(self, original_yaml, temp_yaml, fraction):
        """Create a temporary YAML using a text file of subset paths (Massive I/O optimization)"""
        # Ensure random seed is set before sampling
        random.seed(self.seed)
        
        # Load original YAML
        with open(original_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Get all image paths from train split
        train_dir = self.root_dir / "images" / "train"
        all_images = list(train_dir.glob("*.jpg"))
        
        # Sort first to ensure consistent ordering before random sampling
        all_images.sort()
        
        # Create subset with the instance seed
        subset_size = int(len(all_images) * fraction)
        random.shuffle(all_images)
        subset_images = all_images[:subset_size]
        
        print(f"   Creating training subset: {len(subset_images)}/{len(all_images)} images ({fraction*100:.0f}%)")
        
        # Write subset paths to a .txt file
        subset_txt = self.results_dir / f"train_subset_f{fraction}_s{self.seed}.txt"
        with open(subset_txt, 'w') as f:
            for img_path in subset_images:
                f.write(f"{img_path.absolute()}\n")
                
        # Update YAML to point to the .txt file instead of a folder
        # Ensure absolute paths so YOLO doesn't get confused
        yaml_data['path'] = str(self.root_dir.absolute())
        yaml_data['train'] = str(subset_txt.absolute())
        
        # Ensure validation and test paths are correctly relative to the new absolute root
        if 'val' in yaml_data and not str(yaml_data['val']).startswith('/'):
            yaml_data['val'] = f"images/val" 
        if 'test' in yaml_data and not str(yaml_data['test']).startswith('/'):
            yaml_data['test'] = f"images/test"
        
        # Save temporary YAML
        with open(temp_yaml, 'w') as f:
            yaml.dump(yaml_data, f)
            
        print(f"   Fast subset YAML created. Paths written to {subset_txt.name}")
        return len(subset_images)

    def _get_dataset_size(self, dataset_fraction):
        """Get the actual dataset size for a given fraction"""
        train_dir = self.root_dir / "images" / "train"
        all_images = list(train_dir.glob("*.jpg"))
        return int(len(all_images) * dataset_fraction)
        
    def measure_model_complexity(self, model, input_size):
        """Measure model FLOPs and parameters with proper memory management"""
        input_tensor = torch.randn(1, 3, input_size, input_size)
        device = next(model.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Prevent PyTorch from building a VRAM-hogging computational graph
        model.eval()
        
        # Function to recursively clear THOP buffers from all modules
        def clear_thop_buffers(module):
            buffers_to_remove = ['total_ops', 'total_params']
            for buffer_name in buffers_to_remove:
                if hasattr(module, buffer_name):
                    delattr(module, buffer_name)
            for child in module.children():
                clear_thop_buffers(child)
        
        with torch.no_grad():
            try:
                clear_thop_buffers(model.model)
                flops, params = profile(model.model, inputs=(input_tensor,), verbose=False)
                print(f"   Complexity measured with THOP")
            except Exception as e:
                print(f"   THOP failed: {e}. Using fallback estimation.")
                total_params = sum(p.numel() for p in model.model.parameters())
                flops = total_params * input_size * input_size * 2
                params = total_params
                
        # Aggressive cleanup before training begins
        del input_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return flops, params
    
    def measure_inference_performance(self, model, input_size, n_iterations=50):
            """Measure inference time and throughput with FP16 and sync"""
            
            # --- FIX: Extract the raw PyTorch model to bypass Ultralytics overhead ---
            # This prevents the context manager crash and measures pure NN hardware speed
            pytorch_model = model.model
            pytorch_model.eval() 
            
            input_tensor = torch.rand(1, 3, input_size, input_size)
            device = next(pytorch_model.parameters()).device
            input_tensor = input_tensor.to(device)
            
            # Use half precision for realistic deployment benchmarking on T4/A2
            if device.type == 'cuda':
                pytorch_model.half()
                input_tensor = input_tensor.half()
                
            with torch.no_grad(): 
                # Warm up
                for _ in range(10):
                    _ = pytorch_model(input_tensor)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                
                # Time inference
                times = []
                for _ in range(n_iterations):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                        start = time.time()
                        _ = pytorch_model(input_tensor)
                        torch.cuda.synchronize()
                        end = time.time()
                    else:
                        start = time.time()
                        _ = pytorch_model(input_tensor)
                        end = time.time()
                    
                    times.append(end - start)
            
            # Clean up FP16 weights to prevent memory fragmentation
            if device.type == 'cuda':
                pytorch_model.float()
                del input_tensor
                torch.cuda.empty_cache()
                
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1 / avg_time
            
            return avg_time, std_time, fps

    def is_experiment_completed(self, model_variant, dataset_fraction, resolution):
        """Check if this experiment combination has already been completed FOR THIS SEED"""
        for result in self.results:
            if (result['model_variant'] == model_variant and 
                result['dataset_fraction'] == dataset_fraction and 
                result['resolution'] == resolution and
                result.get('seed', 42) == self.seed): # Now specific to the seed
                return True
        return False
    
    def run_experiment(self, seed, model_variant, dataset_fraction, resolution):
        """Run a single experiment with specific parameters"""
        self.seed = seed
        self._set_seeds()
        print(f"\nRunning Experiment:")
        print(f"   Model: {model_variant}")
        print(f"   Dataset: {dataset_fraction*100:.0f}%")
        print(f"   Resolution: {resolution}px")
        print(f"   Seed: {self.seed}")
        exp_name = f"{model_variant.split('.')[0]}_f{dataset_fraction}_r{resolution}_s{self.seed}"
        
        # Check if already completed for this specific seed
        if self.is_experiment_completed(model_variant, dataset_fraction, resolution):
            print(f"   Experiment already completed for seed {self.seed}, skipping...")
            return None
        
        # Load model
        model = YOLO(model_variant)
        
        # Measure model complexity
        flops, params = self.measure_model_complexity(model, resolution)
        
        # Calculate conservative physical batch size
        batch_size = 32
        
        # Handle dataset fraction by modifying the YAML file temporarily
        original_yaml = self.root_dir / self.yaml_file
        temp_yaml = self.results_dir / f"temp_dataset_f{dataset_fraction}_s{self.seed}.yaml"
        subset_txt = self.results_dir / f"train_subset_f{dataset_fraction}_s{self.seed}.txt"
        
        if dataset_fraction < 1.0:
            self._create_subset_yaml(original_yaml, temp_yaml, dataset_fraction)
            data_path = str(temp_yaml)
        else:
            data_path = str(original_yaml)
            print(f"   Using full dataset")
        
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            compile_mode = "reduce-overhead"
        elif torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple Silicon (MPS)"
            compile_mode = False
        else:
            device = "cpu"
            device_name = "CPU"
            compile_mode = False
        print(device)
        print(f"   Using device: {device_name}")
        print(f"   Training for {self.epochs} epochs with physical batch size {batch_size} (Mathematical batch=64)...")
        if compile_mode:
            print(f"   Native compilation enabled: {compile_mode}")
        
        wandb.init(
            project="YOLO-Scaling-Laws", # All 60 runs will go into this dashboard
            name=exp_name,
            config={
                "model_variant": model_variant,
                "dataset_fraction": dataset_fraction,
                "resolution": resolution,
                "seed": self.seed,
                "epochs": self.epochs,
                "physical_batch_size": batch_size,
                "mathematical_batch_size": 64,
                "optimizer": "AdamW"
            }
        )
        train_results = model.train(
            data=data_path,
            epochs=self.epochs,
            imgsz=resolution,
            device=device,
            batch=batch_size,   # Fixed physical batch to prevent OOM
            nbs=64,             # Enforce identical mathematical batch size across all models
            compile=compile_mode,
            seed=self.seed,     # Ensure YOLO engine deterministic behavior
            deterministic=True, # Force CuDNN deterministic algorithms inside YOLO
            rect=True,
            verbose=False,
            save=True,
            plots=True,
            val=True,
            patience=self.config['training']['patience'],
            lr0=self.config['training']['lr0'],
            lrf=self.config['training']['lrf'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_epochs=self.config['training']['warmup_epochs'],
            warmup_momentum=self.config['training']['warmup_momentum'],
            warmup_bias_lr=self.config['training']['warmup_bias_lr'],
            amp=True,
            project="YOLO-Scaling-Laws",
            name=exp_name,
            optimizer='AdamW',  # Force consistent optimizer
            cos_lr=True,
        )
        
        best_weights_path = Path(train_results.save_dir) / "weights" / "best.pt"
        last_weights_path = Path(train_results.save_dir) / "weights" / "last.pt"
        
        if not best_weights_path.exists():
            print(f"   WARNING: {best_weights_path} not found! Using final memory weights.")
            best_model = model # Fallback
        else:
            print("   Loading best checkpoint from disk...")
            best_model = YOLO(str(best_weights_path))

        # Clean up temporary files
        if dataset_fraction < 1.0:
            if temp_yaml.exists(): temp_yaml.unlink()
            if subset_txt.exists(): subset_txt.unlink()
        
        # Evaluate on validation set
        print("   Evaluating on validation set...")
        val_results = best_model.val(
            data=str(self.root_dir / self.yaml_file),
            imgsz=resolution,
            verbose=False,
            split="test"
        )
        
        # Measure inference performance
        print("   Measuring inference performance...")
        avg_time, std_time, fps = self.measure_inference_performance(best_model, resolution)
        
        # Measure GPU memory usage if available
        gpu_memory_used = 0
        if device == "cuda":
            gpu_memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            torch.cuda.reset_peak_memory_stats()
        
        # Extract metrics
        metrics = val_results.results_dict if hasattr(val_results, 'results_dict') else {}
        if wandb.run is not None:
            wandb.run.summary["efficiency/flops"] = flops
            wandb.run.summary["efficiency/parameters"] = params
            wandb.run.summary["efficiency/inference_fps"] = fps
            wandb.run.summary["efficiency/inference_time_ms"] = avg_time * 1000
        
        # Close the W&B run so the next experiment starts fresh
        wandb.finish()
        result = {
            'seed': self.seed,
            'model_variant': model_variant,
            'dataset_fraction': dataset_fraction,
            'resolution': resolution,
            'dataset_size': self._get_dataset_size(dataset_fraction),
            'epochs': self.epochs,
            'physical_batch_size': batch_size,
            'mathematical_batch_size': 64, # Fixed via nbs
            'compile_mode': str(compile_mode),
            'device': device,
            'device_name': device_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            
            # Quality metrics
            'mAP50': metrics.get('metrics/mAP50(B)', 0.0),
            'mAP50_95': metrics.get('metrics/mAP50-95(B)', 0.0),
            'precision': metrics.get('metrics/precision(B)', 0.0),
            'recall': metrics.get('metrics/recall(B)', 0.0),
            'f1': metrics.get('metrics/f1(B)', 0.0),
            
            # Efficiency metrics
            'flops': flops,
            'params': params,
            'inference_time_ms': avg_time * 1000,
            'inference_std_ms': std_time * 1000,
            'fps': fps,
            'gpu_memory_mb': gpu_memory_used,
            
            # Training metrics
            'train_loss': train_results.results_dict.get('train/box_loss', 0.0) if hasattr(train_results, 'results_dict') else 0.0,
            'val_loss': train_results.results_dict.get('val/box_loss', 0.0) if hasattr(train_results, 'results_dict') else 0.0,
        }
        
        self.results.append(result)
        print(f"   mAP@0.5: {result['mAP50']:.3f} | FPS: {result['fps']:.1f} | Mem: {gpu_memory_used:.0f}MB")
        
        # Save results immediately
        self._save_results()

        if best_weights_path.exists():
            best_weights_path.unlink()
        if last_weights_path.exists():
            last_weights_path.unlink()
        
        # --- Aggressive VRAM cleanup to prevent OOM on next iteration ---
        del model
        del train_results
        del val_results
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return result

def main():
    parser = argparse.ArgumentParser(description='YOLO Scaling Law Study - Individual Experiment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    
    # Experiment parameters
    parser.add_argument('--dataset_fraction', type=float, required=True, help='Dataset fraction (0.1, 0.25, 0.5, 1.0)')
    parser.add_argument('--model_variant', type=str, required=True, help='Model variant (yolo11n.pt, ...)')
    parser.add_argument('--resolution', type=int, required=True, help='Input resolution (416, 640, 1280)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for experiment variance')
    
    # Utility commands
    parser.add_argument('--list_combinations', action='store_true', help='List all possible experiment combinations')
    parser.add_argument('--show_progress', action='store_true', help='Show current progress')
    
    args = parser.parse_args()
    
    # Initialize study
    study = ScalingLawStudy(args.config, args.seed)
    
    # Handle utility commands
    if args.list_combinations:
        # Note: Added logic locally assuming combinations logic remains unchanged. 
        return
        
    if args.show_progress:
        study.show_progress()
        return
    
    # Run experiment
    study.run_experiment(args.seed, args.model_variant, args.dataset_fraction, args.resolution)

if __name__ == "__main__":
    main()