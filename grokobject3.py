import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Define IoU Calculation Functions
def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x_center, y_center, width, height]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to top-left and bottom-right coordinates
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2
    
    # Intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Intersection area
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def load_labels(label_path):
    """Load YOLO-format labels from a text file."""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.strip().split())) for line in f]
    return labels

def evaluate_iou(pred_dir, gt_dir):
    """Calculate average IoU for all images."""
    ious = []
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]
    
    for pred_file in pred_files:
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, pred_file)
        
        if not os.path.exists(gt_path):
            continue
            
        pred_labels = load_labels(pred_path)
        gt_labels = load_labels(gt_path)
        
        # Match predictions to ground truth (simplified: assume same order)
        for pred, gt in zip(pred_labels, gt_labels):
            if len(pred) < 5 or len(gt) < 5:
                continue
            pred_box = pred[1:5]  # x_center, y_center, width, height
            gt_box = gt[1:5]
            iou = calculate_iou(pred_box, gt_box)
            ious.append(iou)
    
    return np.mean(ious) if ious else 0

# Step 2: Create Custom Hyperparameters File
hyp_content = """lr0: 0.01
lrf: 0.2
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
"""

hyp_path = r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\yolov5\hyp.custom.yaml'
os.makedirs(os.path.dirname(hyp_path), exist_ok=True)  # Create yolov5 directory if it doesn't exist
with open(hyp_path, 'w') as f:
    f.write(hyp_content)

# Step 3: Run Baseline YOLOv5 Evaluation
def run_baseline():
    cmd = [
        'python', 'val.py',
        '--weights', 'yolov5s.pt',
        '--data', r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\VisDrone2019-DET\data.yaml',
        '--img', '640',
        '--task', 'val',
        '--save-txt',
        '--save-conf',
        '--project', r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\runs\baseline',
        '--name', 'exp'
    ]
    subprocess.run(cmd, check=True)
    
    # Calculate IoU
    pred_dir = r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\runs\baseline\exp\labels'
    gt_dir = r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\VisDrone2019-DET\valid\labels'
    baseline_iou = evaluate_iou(pred_dir, gt_dir)
    print(f"Baseline IoU: {baseline_iou:.4f}")
    return baseline_iou

# Step 4: Train Improved YOLOv5 Model
def train_improved():
    cmd = [
        'python', 'train.py',
        '--img', '640',
        '--batch', '16',
        '--epochs', '50',
        '--data', r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\VisDrone2019-DET\data.yaml',
        '--weights', 'yolov5m.pt',
        '--hyp', hyp_path,
        '--project', r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\runs\improved',
        '--name', 'exp'
    ]
    subprocess.run(cmd, check=True)

# Step 5: Run Improved Model Evaluation
def run_improved():
    cmd = [
        'python', 'val.py',
        '--weights', r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\runs\improved\exp\weights\best.pt',
        '--data', r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\VisDrone2019-DET\data.yaml',
        '--img', '640',
        '--task', 'val',
        '--save-txt',
        '--save-conf',
        '--project', r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\runs\improved',
        '--name', 'exp'
    ]
    subprocess.run(cmd, check=True)
    
    # Calculate IoU
    pred_dir = r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\runs\improved\exp\labels'
    gt_dir = r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\VisDrone2019-DET\valid\labels'
    improved_iou = evaluate_iou(pred_dir, gt_dir)
    print(f"Improved IoU: {improved_iou:.4f}")
    return improved_iou

# Step 6: Plot IoU Comparison
def plot_comparison(baseline_iou, improved_iou):
    models = ['Baseline', 'Improved']
    ious = [baseline_iou, improved_iou]
    
    plt.figure(figsize=(8, 6))
    plt.bar(models, ious, color=['blue', 'green'])
    plt.xlabel('Model')
    plt.ylabel('IoU Accuracy')
    plt.title('IoU Accuracy Comparison: Baseline vs Improved YOLOv5')
    plt.ylim(0, 1)
    for i, v in enumerate(ious):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    plt.savefig(r'C:\Users\Lenovo\Dropbox\PC\Desktop\objectdetectiongrok\iou_comparison.png')
    plt.show()

# Main Execution
def main():
    print("Evaluating Baseline Model...")
    baseline_iou = run_baseline()
    
    print("Training Improved Model...")
    train_improved()
    
    print("Evaluating Improved Model...")
    improved_iou = run_improved()
    
    print("Plotting IoU Comparison...")
    plot_comparison(baseline_iou, improved_iou)

if __name__ == "__main__":
    main()