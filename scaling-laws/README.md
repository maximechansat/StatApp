# YOLO Scaling Law Study - Individual Experiment Execution

This framework allows you to run individual experiments with specific parameters for the YOLO scaling law study.

## Study Parameters

### **Dataset Fractions:** 0.1, 0.25, 0.5, 1.0 (10%, 25%, 50%, 100%)
### **Model Variants:** yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
### **Resolutions:** 416, 640, 1280 pixels

**Total Combinations:** 4 × 5 × 3 = **60 experiments**

## Quick Start

### **1. Configure Dataset Path**
Edit `config.yaml` to set your dataset path:
```yaml
dataset:
  root_dir: "/path/to/your/dataset"
```

### **2. Run Individual Experiments**
```bash
# Example: Run YOLO-nano with 25% dataset at 640px resolution
python scaling_law_study.py \
    --dataset_fraction 0.25 \
    --model_variant yolo11n.pt \
    --resolution 640
```

### **3. Check Progress**
```bash
# List all possible combinations
python scaling_law_study.py --list_combinations

# Show current progress
python scaling_law_study.py --show_progress
```

## Usage Examples

### **Run Specific Experiments:**
```bash
# Test different dataset sizes with YOLO-nano at 640px
python scaling_law_study.py --dataset_fraction 0.1 --model_variant yolo11n.pt --resolution 640
python scaling_law_study.py --dataset_fraction 0.25 --model_variant yolo11n.pt --resolution 640
python scaling_law_study.py --dataset_fraction 0.5 --model_variant yolo11n.pt --resolution 640
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11n.pt --resolution 640

# Test different model sizes with full dataset at 640px
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11n.pt --resolution 640
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11s.pt --resolution 640
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11m.pt --resolution 640
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11l.pt --resolution 640
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11x.pt --resolution 640

# Test different resolutions with YOLO-nano and full dataset
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11n.pt --resolution 416
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11n.pt --resolution 640
python scaling_law_study.py --dataset_fraction 1.0 --model_variant yolo11n.pt --resolution 1280
```

## Results

### **Output Files:**
```
scaling_results/
├── results.json          # All experiment results
├── results.csv           # CSV format for analysis
├── yolo11n_frac0.25_res640.pt  # Trained models (if save_models: true)
└── yolo11s_frac1.0_res640.pt
```

### **Result Format:**
Each experiment saves:
- **Quality Metrics:** mAP@0.5, mAP@0.5:0.95, precision, recall, F1
- **Efficiency Metrics:** FLOPs, parameters, model size, inference time, FPS, GPU memory usage
- **Training Metrics:** training loss, validation loss
- **Metadata:** timestamp, dataset size, batch size, epochs, device info

## Data Transforms and Augmentation

### **How Data Transforms Work:**

The scaling law study uses **YOLO's built-in data loading and augmentation system**, which provides:

1. **Automatic Resolution Handling**: YOLO automatically resizes images to the specified resolution (416, 640, 1280px)
2. **Built-in Augmentation**: YOLO applies standard augmentations like:
   - Mosaic augmentation
   - Random horizontal/vertical flips
   - Color space augmentation (HSV)
   - Random scaling and translation
   - Mixup and CutMix (if enabled)

3. **Dynamic Batch Sizing**: Batch size automatically scales with resolution to optimize memory usage

4. **Proper Bounding Box Handling**: All augmentations properly transform bounding box coordinates

### **Dataset Fraction Handling:**
- For fractions < 1.0, the system creates temporary subset datasets
- Images are randomly sampled to maintain class distribution
- Temporary files are cleaned up after each experiment

### **Device Support:**
- **CUDA**: Automatic detection with proper GPU synchronization
- **MPS**: Apple Silicon support (M1/M2/M3 chips)
- **CPU**: Fallback for systems without GPU support
- **Accurate Timing**: Uses `torch.cuda.synchronize()` for precise GPU measurements
- **Memory Tracking**: Monitors GPU memory usage on CUDA devices

## Configuration

### **config.yaml Structure:**
```yaml
dataset:
  root_dir: "/path/to/dataset"
  yaml_file: "solar_panel_dataset.yaml"

study:
  dataset_fractions: [0.1, 0.25, 0.5, 1.0]
  model_variants: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
  resolutions: [416, 640, 1280]
  epochs: 20

training:
  batch_size_scale: 64
  patience: 10
  lr0: 0.01
  # ... other training parameters

results:
  output_dir: "scaling_results"
  save_models: true
  save_plots: true
```

## Advanced Usage

### **Custom Configuration:**
```bash
# Use custom config file
python scaling_law_study.py --config my_config.yaml --dataset_fraction 0.5 --model_variant yolo11s.pt --resolution 640
```

### **Batch Execution Script:**
Create a script to run multiple experiments:
```bash
#!/bin/bash
# run_all_experiments.sh

# Run all combinations
for dataset in 0.1 0.25 0.5 1.0; do
    for model in yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt; do
        for resolution in 416 640 1280; do
            echo "Running: $model, $dataset, $resolution"
            python scaling_law_study.py \
                --dataset_fraction $dataset \
                --model_variant $model \
                --resolution $resolution
        done
    done
done
```

## Progress Tracking

### **Check Completion Status:**
```bash
python scaling_law_study.py --show_progress
```

**Sample Output:**
```
📊 Progress Report
   Completed: 15/60 (25.0%)
   Best mAP@0.5: 0.742
   Best FPS: 45.3

   Dataset fractions: 2/4
   Model variants: 3/5
   Resolutions: 3/3
```

### **List All Combinations:**
```bash
python scaling_law_study.py --list_combinations
```
