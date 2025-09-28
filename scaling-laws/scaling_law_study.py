# YOLO Scaling Law Study - Individual Experiment Execution
# Run single experiments with specific parameters

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Subset
from ultralytics import YOLO
from thop import profile
import random
import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('../')
from src.dataset import SolarPanelDataset, collate_fn

class ScalingLawStudy:
    """
    Individual experiment execution for YOLO scaling law study.
    
    Run single experiments with specific parameters:
    - dataset_fraction: 0.1, 0.25, 0.5, 1.0
    - model_variant: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    - resolution: 416, 640, 1280
    """
    
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.root_dir = Path(self.config['dataset']['root_dir'])
        self.yaml_file = self.config['dataset']['yaml_file']
        self.results_dir = Path(self.config['results']['output_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.epochs = self.config['study']['epochs']
        self.batch_scale = self.config['training']['batch_size_scale']
        
        # Set seeds for reproducibility
        self._set_seeds()
        
        # Load existing results
        self._load_existing_results()
        
    def _set_seeds(self):
        """Set all random seeds for reproducibility"""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(42)
        np.random.seed(42)
        
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
        """Save results to JSON"""
        results_file = self.results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as CSV for easy analysis
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.results_dir / "results.csv"
            df.to_csv(csv_file, index=False)
        
        print(f"Results saved ({len(self.results)} total experiments)")
    
    def _create_subset_yaml(self, original_yaml, temp_yaml, fraction):
        """Create a temporary YAML file with a subset of training data and complete val/test sets"""
        import yaml
        import shutil
        
        # Ensure random seed is set before sampling
        random.seed(42)
        
        # Load original YAML
        with open(original_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Get all image paths from train split (canonical structure)
        train_dir = self.root_dir / "images" / "train"
        all_images = list(train_dir.glob("*.jpg"))
        
        # Sort first to ensure consistent ordering across different systems
        all_images.sort()
        
        # Create subset with fixed seed
        subset_size = int(len(all_images) * fraction)
        random.shuffle(all_images)
        subset_images = all_images[:subset_size]
        
        print(f"   Creating training subset: {len(subset_images)}/{len(all_images)} images ({fraction*100:.0f}%)")
        
        # Create temporary directory structure (complete canonical YOLO structure)
        temp_base = self.results_dir / "temp_images"
        
        # All splits
        splits = ['train', 'val', 'test']
        temp_dirs = {}
        
        for split in splits:
            temp_dirs[split] = {
                'images': temp_base / "images" / split,
                'labels': temp_base / "labels" / split
            }
            # Create directories
            temp_dirs[split]['images'].mkdir(parents=True, exist_ok=True)
            temp_dirs[split]['labels'].mkdir(parents=True, exist_ok=True)
        
        # Copy TRAINING subset (images + labels)
        for img_path in subset_images:
            dest_img = temp_dirs['train']['images'] / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
        
        # Copy corresponding training labels
        original_train_labels_dir = self.root_dir / "labels" / "train"
        train_labels_copied = 0
        train_labels_missing = 0

        for img_path in subset_images:
            label_name = img_path.stem + ".txt"
            original_label = original_train_labels_dir / label_name
            dest_label = temp_dirs['train']['labels'] / label_name
            
            if original_label.exists():
                shutil.copy2(original_label, dest_label)
                train_labels_copied += 1
            else:
                train_labels_missing += 1
                print(f"     Missing label: {label_name}")

        print(f"   Training: {len(subset_images)} images, {train_labels_copied} labels ({train_labels_missing} missing)")
        # Copy COMPLETE validation and test sets (images + labels)
        for split in ['val', 'test']:
            original_img_dir = self.root_dir / "images" / split
            original_label_dir = self.root_dir / "labels" / split
            
            images_copied = 0
            labels_copied = 0
            
            # Copy all images for this split
            if original_img_dir.exists():
                for img_path in original_img_dir.glob("*.jpg"):
                    dest_img = temp_dirs[split]['images'] / img_path.name
                    if not dest_img.exists():
                        shutil.copy2(img_path, dest_img)
                        images_copied += 1
            
            # Copy all labels for this split
            if original_label_dir.exists():
                for label_path in original_label_dir.glob("*.txt"):
                    dest_label = temp_dirs[split]['labels'] / label_path.name
                    if not dest_label.exists():
                        shutil.copy2(label_path, dest_label)
                        labels_copied += 1
            
            if images_copied > 0 or labels_copied > 0:
                print(f"   {split.capitalize()}: {images_copied} images, {labels_copied} labels (complete)")
        
        # Update YAML to point to temporary dataset (canonical structure)
        yaml_data['path'] = str(temp_base)
        yaml_data['train'] = "images/train"
        yaml_data['val'] = "images/val"
        if 'test' in yaml_data:
            yaml_data['test'] = "images/test"
        
        # Save temporary YAML
        with open(temp_yaml, 'w') as f:
            yaml.dump(yaml_data, f)
        
        # Verify the complete structure
        summary = {}
        for split in splits:
            images_count = len(list(temp_dirs[split]['images'].glob("*.jpg")))
            labels_count = len(list(temp_dirs[split]['labels'].glob("*.txt")))
            summary[split] = {'images': images_count, 'labels': labels_count}
        
        print(f"   Complete temp dataset created:")
        print(f"     Train: {summary['train']['images']} images, {summary['train']['labels']} labels (subset)")
        print(f"     Val: {summary['val']['images']} images, {summary['val']['labels']} labels (complete)")
        print(f"     Test: {summary['test']['images']} images, {summary['test']['labels']} labels (complete)")
        
        # Warn if there are significant mismatches
        for split in splits:
            images = summary[split]['images']
            labels = summary[split]['labels']
            if images > 0 and labels < images * 0.9:
                print(f"   WARNING: {split} has only {labels}/{images} labels!")
        
        return summary

    def _get_dataset_size(self, dataset_fraction):
        """Get the actual dataset size for a given fraction"""
        train_dir = self.root_dir / "images" / "train"
        all_images = list(train_dir.glob("*.jpg"))
        return int(len(all_images) * dataset_fraction)
        
    def measure_model_complexity(self, model, input_size):
        """Measure model FLOPs and parameters with multiple fallback methods"""
        input_tensor = torch.randn(1, 3, input_size, input_size)
        
        # Move to same device as model
        device = next(model.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Function to recursively clear THOP buffers from all modules
        def clear_thop_buffers(module):
            buffers_to_remove = ['total_ops', 'total_params']
            for buffer_name in buffers_to_remove:
                if hasattr(module, buffer_name):
                    delattr(module, buffer_name)
            for child in module.children():
                clear_thop_buffers(child)
        
        # Method 1: Try with the underlying PyTorch model
        try:
            clear_thop_buffers(model.model)
            flops, params = profile(model.model, inputs=(input_tensor,), verbose=False)
            print(f"   Complexity measured with THOP (method 1)")
            return flops, params
        except Exception as e:
            print(f"   THOP method 1 failed: {e}")
        
        # Method 2: Try creating a fresh copy of the model
        try:
            # Get the model's state dict
            model_copy = type(model.model)()
            model_copy.load_state_dict(model.model.state_dict())
            model_copy = model_copy.to(device)
            
            flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
            print(f"   Complexity measured with THOP (method 2 - model copy)")
            return flops, params
        except Exception as e:
            print(f"   THOP method 2 failed: {e}")
        
        # Method 3: Try with a completely fresh model instance
        try:
            from ultralytics import YOLO
            fresh_model = YOLO(model.ckpt_path if hasattr(model, 'ckpt_path') else 'yolo11n.pt')
            clear_thop_buffers(fresh_model.model)
            
            flops, params = profile(fresh_model.model, inputs=(input_tensor,), verbose=False)
            print(f"   Complexity measured with THOP (method 3 - fresh model)")
            return flops, params
        except Exception as e:
            print(f"   THOP method 3 failed: {e}")
        
        # Fallback: Manual parameter counting + FLOP estimation
        print(f"   All THOP methods failed, using fallback estimation")
        total_params = sum(p.numel() for p in model.model.parameters())
        
        # Better FLOP estimation based on typical YOLO architectures
        # This accounts for the fact that not all parameters contribute equally to FLOPs
        # Rough estimation: each parameter contributes ~2 operations per pixel
        estimated_flops = total_params * input_size * input_size * 2
        
        print(f"   Estimated: FLOPs={estimated_flops:,}, Parameters={total_params:,}")
        return estimated_flops, total_params
    
    def measure_inference_performance(self, model, input_size, n_iterations=50):
        """Measure inference time and throughput with proper GPU synchronization"""
        # Create test input with proper normalization (0-1 range for YOLO)
        input_tensor = torch.rand(1, 3, input_size, input_size)  # Use rand() instead of randn()
        
        # Move to same device as model
        device = next(model.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warm up with proper synchronization
        model.eval()  # Ensure model is in eval mode
        with torch.no_grad():  # Disable gradients for inference
            for _ in range(10):
                _ = model(input_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Time inference with proper GPU synchronization
        times = []
        with torch.no_grad():  # Disable gradients for inference timing
            for _ in range(n_iterations):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    start = time.time()
                    _ = model(input_tensor)
                    torch.cuda.synchronize()
                    end = time.time()
                else:
                    # For CPU/MPS, no synchronization needed
                    start = time.time()
                    _ = model(input_tensor)
                    end = time.time()
                
                times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1 / avg_time
        
        return avg_time, std_time, fps

    def is_experiment_completed(self, model_variant, dataset_fraction, resolution):
        """Check if this experiment combination has already been completed"""
        for result in self.results:
            if (result['model_variant'] == model_variant and 
                result['dataset_fraction'] == dataset_fraction and 
                result['resolution'] == resolution):
                return True
        return False
    
    def run_experiment(self, model_variant, dataset_fraction, resolution):
        """Run a single experiment with specific parameters"""
        print(f"\nRunning Experiment:")
        print(f"   Model: {model_variant}")
        print(f"   Dataset: {dataset_fraction*100:.0f}%")
        print(f"   Resolution: {resolution}px")
        
        # Check if already completed
        if self.is_experiment_completed(model_variant, dataset_fraction, resolution):
            print(f"   Experiment already completed, skipping...")
            return None
        
        # Load model
        model = YOLO(model_variant)
        
        # Measure model complexity
        flops, params = self.measure_model_complexity(model, resolution)
        
        # Calculate batch size based on resolution
        batch_size = min(32, max(4, self.batch_scale // (resolution // 320)))
        
        # Handle dataset fraction by modifying the YAML file temporarily
        original_yaml = self.root_dir / self.yaml_file
        temp_yaml = self.results_dir / f"temp_dataset_{dataset_fraction}.yaml"
        
        if dataset_fraction < 1.0:
            # Create a temporary YAML file with subset of data
            self._create_subset_yaml(original_yaml, temp_yaml, dataset_fraction)
            data_path = str(temp_yaml)
            print(f"   Using {dataset_fraction*100:.0f}% of dataset (subset created)")
        else:
            data_path = str(original_yaml)
            print(f"   Using full dataset")
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple Silicon (MPS)"
        else:
            device = "cpu"
            device_name = "CPU"
        
        print(f"   Using device: {device_name}")
        print(f"   Training for {self.epochs} epochs with batch size {batch_size}...")
        
        train_results = model.train(
            data=data_path,
            epochs=self.epochs,
            imgsz=resolution,
            device=device,
            batch=batch_size,
            rect=True,
            verbose=False,
            save=False,
            plots=False,
            val=False,
            # Training parameters from config
            patience=self.config['training']['patience'],
            lr0=self.config['training']['lr0'],
            lrf=self.config['training']['lrf'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_epochs=self.config['training']['warmup_epochs'],
            warmup_momentum=self.config['training']['warmup_momentum'],
            warmup_bias_lr=self.config['training']['warmup_bias_lr'],
        )
        
        # Clean up temporary YAML file
        if dataset_fraction < 1.0 and temp_yaml.exists():
            temp_yaml.unlink()
        
        # Evaluate on validation set
        print("   Evaluating on validation set...")
        val_results = model.val(
            data=str(self.root_dir / self.yaml_file),
            imgsz=resolution,
            verbose=False
        )
        
        # Measure inference performance
        print("   Measuring inference performance...")
        avg_time, std_time, fps = self.measure_inference_performance(model, resolution)
        
        # Measure GPU memory usage if available
        gpu_memory_used = 0
        if device == "cuda":
            gpu_memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            torch.cuda.reset_peak_memory_stats()
        
        # Save model if configured
        if self.config['results']['save_models']:
            model_name = f"{model_variant.split('.')[0]}_frac{dataset_fraction}_res{resolution}"
            model_path = self.results_dir / f"{model_name}.pt"
            model.save(model_path)
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
        else:
            model_size_mb = 0
        
        # Extract metrics
        metrics = val_results.results_dict if hasattr(val_results, 'results_dict') else {}
        
        result = {
            'model_variant': model_variant,
            'dataset_fraction': dataset_fraction,
            'resolution': resolution,
            'dataset_size': self._get_dataset_size(dataset_fraction),
            'epochs': self.epochs,
            'batch_size': batch_size,
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
            'model_size_mb': model_size_mb,
            'inference_time_ms': avg_time * 1000,
            'inference_std_ms': std_time * 1000,
            'fps': fps,
            'gpu_memory_mb': gpu_memory_used,
            
            # Training metrics
            'train_loss': train_results.results_dict.get('train/box_loss', 0.0) if hasattr(train_results, 'results_dict') else 0.0,
            'val_loss': train_results.results_dict.get('val/box_loss', 0.0) if hasattr(train_results, 'results_dict') else 0.0,
        }
        
        self.results.append(result)
        print(f"   mAP@0.5: {result['mAP50']:.3f} | FPS: {result['fps']:.1f} | Params: {result['params']:,}")
        
        # Save results immediately
        self._save_results()
        
        return result
    
    def list_all_combinations(self):
        """List all possible experiment combinations"""
        combinations = []
        for dataset_fraction in self.config['study']['dataset_fractions']:
            for model_variant in self.config['study']['model_variants']:
                for resolution in self.config['study']['resolutions']:
                    combinations.append({
                        'dataset_fraction': dataset_fraction,
                        'model_variant': model_variant,
                        'resolution': resolution
                    })
        return combinations
    
    def show_progress(self):
        """Show current progress"""
        if not self.results:
            print("No experiments completed yet.")
            return
        
        df = pd.DataFrame(self.results)
        total_combinations = len(self.list_all_combinations())
        completed = len(df)
        completion_rate = completed / total_combinations * 100
        
        print(f"\nProgress Report")
        print(f"   Completed: {completed}/{total_combinations} ({completion_rate:.1f}%)")
        print(f"   Best mAP@0.5: {df['mAP50'].max():.3f}")
        print(f"   Best FPS: {df['fps'].max():.1f}")
        
        # Show completion by dimension
        print(f"\n   Dataset fractions: {df['dataset_fraction'].nunique()}/{len(self.config['study']['dataset_fractions'])}")
        print(f"   Model variants: {df['model_variant'].nunique()}/{len(self.config['study']['model_variants'])}")
        print(f"   Resolutions: {df['resolution'].nunique()}/{len(self.config['study']['resolutions'])}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Scaling Law Study - Individual Experiment')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    # Experiment parameters
    parser.add_argument('--dataset_fraction', type=float, required=True,
                       help='Dataset fraction (0.1, 0.25, 0.5, 1.0)')
    parser.add_argument('--model_variant', type=str, required=True,
                       help='Model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--resolution', type=int, required=True,
                       help='Input resolution (416, 640, 1280)')
    
    # Utility commands
    parser.add_argument('--list_combinations', action='store_true',
                       help='List all possible experiment combinations')
    parser.add_argument('--show_progress', action='store_true',
                       help='Show current progress')
    
    args = parser.parse_args()
    
    # Initialize study
    study = ScalingLawStudy(args.config)
    
    # Handle utility commands
    if args.list_combinations:
        combinations = study.list_all_combinations()
        print(f"📋 All {len(combinations)} possible combinations:")
        for i, combo in enumerate(combinations, 1):
            print(f"   {i:2d}. {combo['model_variant']} | {combo['dataset_fraction']*100:4.0f}% | {combo['resolution']}px")
        return
    
    if args.show_progress:
        study.show_progress()
        return
    
    # Validate parameters
    if args.dataset_fraction not in study.config['study']['dataset_fractions']:
        print(f"Invalid dataset_fraction: {args.dataset_fraction}")
        print(f"   Valid options: {study.config['study']['dataset_fractions']}")
        return
    
    if args.model_variant not in study.config['study']['model_variants']:
        print(f"Invalid model_variant: {args.model_variant}")
        print(f"   Valid options: {study.config['study']['model_variants']}")
        return
    
    if args.resolution not in study.config['study']['resolutions']:
        print(f"Invalid resolution: {args.resolution}")
        print(f"   Valid options: {study.config['study']['resolutions']}")
        return
    
    # Run experiment
    study.run_experiment(args.model_variant, args.dataset_fraction, args.resolution)

if __name__ == "__main__":
    main()