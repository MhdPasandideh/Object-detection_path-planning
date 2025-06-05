import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.ops import nms

# Custom KITTI Dataset Class
class KITTIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])  # Assuming PNG images
        self.labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])  # Assuming TXT labels
        
        # Validate that images and labels correspond
        assert len(self.images) == len(self.labels), f"Mismatch between number of images ({len(self.images)}) and labels ({len(self.labels)})"
        for img, lbl in zip(self.images, self.labels):
            assert img.split('.')[0] == lbl.split('.')[0], f"Mismatch: {img} and {lbl} do not correspond"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img)

        # Load labels (KITTI format: type, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry)
        label_path = os.path.join(self.label_dir, self.labels[idx])
        boxes = []
        labels = []
        valid_classes = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Roadworks': 4, 'TrafficLight': 5}
        
        with open(label_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if not data or data[0] == 'DontCare':
                    continue
                if data[0] not in valid_classes:
                    continue
                try:
                    bbox = [float(data[4]), float(data[5]), float(data[6]), float(data[7])]
                    class_id = valid_classes[data[0]]
                    boxes.append(bbox)
                    labels.append(class_id)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing label file {label_path}: {e}")
                    continue

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx]), 'img_name': self.images[idx]}

        if self.transform:
            img = self.transform(img)

        return img_tensor, target, img

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, class_idx):
        self.model.eval()
        input_image = input_image.unsqueeze(0).requires_grad_(True)
        outputs = self.model(input_image)
        if len(outputs[0]['scores']) == 0:
            return None  # No predictions to generate Grad-CAM
        score = outputs[0]['scores'][class_idx]
        self.model.zero_grad()
        score.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        cam = F.interpolate(cam.unsqueeze(0), size=input_image.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

# Enhanced Faster R-CNN Model
class EnhancedFasterRCNN:
    def __init__(self, num_classes, iou_threshold=0.5):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def apply_nms(self, boxes, scores):
        keep = nms(boxes, scores, self.iou_threshold)
        return keep

    def train(self, dataloader, num_epochs=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(num_epochs):
            for images, targets, _ in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items() if k != 'img_name'} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

# Function to Calculate IoU
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Function to Calculate AP for a Single Image
def calculate_ap(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, iou_threshold=0.5):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    true_positives = np.zeros(len(pred_boxes))
    false_positives = np.zeros(len(pred_boxes))
    matched = set()
    for i, (p_box, p_label) in enumerate(zip(pred_boxes, pred_labels)):
        best_iou = 0
        best_gt_idx = -1
        for j, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
            if j in matched or p_label != g_label:
                continue
            iou = calculate_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx not in matched:
            true_positives[i] = 1
            matched.add(best_gt_idx)
        else:
            false_positives[i] = 1
    cum_tp = np.cumsum(true_positives)
    cum_fp = np.cumsum(false_positives)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    recalls = cum_tp / (len(gt_boxes) + 1e-6)
    precisions = np.concatenate(([1], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))
    ap = np.trapz(precisions, recalls)
    return ap

# Function to Validate, Save Samples, and Calculate Accuracy
def validate_and_evaluate(image_dir, label_dir, model, num_samples=10):
    dataset = KITTIDataset(image_dir=image_dir, label_dir=label_dir)
    print(f"Dataset size: {len(dataset)} images")
    
    model.model.eval()
    class_names = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Roadworks', 5: 'TrafficLight'}
    miou_list = []
    ap_list = []

    # Initialize Grad-CAM for the first sample
    gradcam = GradCAM(model.model, model.model.backbone.body.layer4[-1].conv3)

    for i in range(min(num_samples, len(dataset))):
        img_tensor, target, img = dataset[i]
        img_tensor_list = [img_tensor.to(model.device)]
        
        # Ground truth
        gt_boxes = target['boxes'].numpy()
        gt_labels = target['labels'].numpy()
        
        # Predictions
        with torch.no_grad():
            pred = model.model(img_tensor_list)[0]
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        keep = model.apply_nms(torch.tensor(pred_boxes), torch.tensor(pred_scores))
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # Calculate mIoU
        ious = []
        for p_box in pred_boxes:
            best_iou = 0
            for g_box in gt_boxes:
                iou = calculate_iou(p_box, g_box)
                best_iou = max(best_iou, iou)
            ious.append(best_iou)
        miou = np.mean(ious) if ious else 0.0
        miou_list.append(miou)

        # Calculate AP
        ap = calculate_ap(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels)
        ap_list.append(ap)

        # Save ground truth image
        img_np = np.array(img)
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, f"{class_names[label]} (GT)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(f'sample_gt_{i}.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # Save predicted image
        img_np = np.array(img)
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_np, f"{class_names[label]} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(f'sample_pred_{i}.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # Generate Grad-CAM for the first sample (i=0)
        if i == 0:
            cam = gradcam.generate(img_tensor.to(model.device), 0)
            if cam is not None:
                img_np = np.array(img)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                superimposed_img = heatmap * 0.4 + img_np / 255.0
                superimposed_img = superimposed_img / superimposed_img.max()
                cv2.imwrite('gradcam_0.jpg', cv2.cvtColor(np.uint8(255 * superimposed_img), cv2.COLOR_RGB2BGR))

        print(f"Sample {i} ({dataset.images[i]}):")
        print(f"  Image shape: {img_tensor.shape}")
        print(f"  Ground truth boxes: {gt_boxes}")
        print(f"  Ground truth labels: {gt_labels}")
        print(f"  Predicted boxes: {pred_boxes}")
        print(f"  Predicted labels: {pred_labels}")
        print(f"  mIoU: {miou:.4f}")
        print(f"  AP: {ap:.4f}")
        print(f"  Saved ground truth image as sample_gt_{i}.jpg")
        print(f"  Saved predicted image as sample_pred_{i}.jpg")
        if i == 0 and cam is not None:
            print("  Saved Grad-CAM visualization as gradcam_0.jpg")

    # Summary
    avg_miou = np.mean(miou_list)
    avg_ap = np.mean(ap_list)
    print(f"\nSummary for {num_samples} samples:")
    print(f"  Average mIoU: {avg_miou:.4f}")
    print(f"  Average AP: {avg_ap:.4f}")

    return miou_list, ap_list, avg_miou, avg_ap

# Main Execution
def main():
    # Specify directories
    image_dir = '/content/image'
    label_dir = '/content/label'

    # Validate directory existence
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Error: One or both directories do not exist: {image_dir}, {label_dir}")
        return

    # Initialize dataset and model
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = KITTIDataset(image_dir=image_dir, label_dir=label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    num_classes = 6  # Background + Car, Pedestrian, Cyclist, Roadworks, TrafficLight
    model = EnhancedFasterRCNN(num_classes=num_classes, iou_threshold=0.5)

    # Train model (optional, can load pre-trained weights instead)
    print("Training model...")
    model.train(dataloader, num_epochs=5)  # Reduced epochs for demonstration

    # Validate and evaluate
    print("\nEvaluating and saving samples...")
    miou_list, ap_list, avg_miou, avg_ap = validate_and_evaluate(image_dir, label_dir, model, num_samples=10)

    # Compare with state-of-the-art
    print("\nState-of-the-Art Comparison (KITTI dataset, mAP@0.5):")
    print(f"  Enhanced Faster R-CNN (ours): {avg_ap:.4f}")
    print("  YOLOv5: ~0.85")
    print("  Mask R-CNN: ~0.80")
    print("  Baseline Faster R-CNN: ~0.75")

if __name__ == "__main__":
    main()