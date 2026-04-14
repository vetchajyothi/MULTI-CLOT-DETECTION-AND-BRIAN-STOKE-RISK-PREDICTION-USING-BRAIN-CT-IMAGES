import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

from src.data_loader import get_data_loaders
from src.models.classification import StrokeClassifier, StrokeTypeClassifier
from src.models.segmentation_detection import UNet
from src.segmentation_dataset import SegmentationDataset

def calculate_dice(preds, targets, smooth=1e-5):
    """Calculate the Dice coefficient (F1 Score) for segmentation."""
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

def calculate_pixel_accuracy(preds, targets):
    """Calculate total pixel accuracy for segmentation."""
    preds = (preds > 0.5).float()
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()

def evaluate_classification(model, test_loader, device, task_name=""):
    model_name = "Stroke Detection" if task_name == "Stroke" else "Stroke Type"
    print(f"\nEvaluating {model_name} Model")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
         for inputs, labels in test_loader:
             inputs, labels = inputs.to(device), labels.to(device)
             outputs = model(inputs)
             _, predicted = torch.max(outputs, 1)
             test_total += labels.size(0)
             test_correct += (predicted == labels).sum().item()
             
    if test_total == 0:
        print(f"No test data found for {task_name}.")
        return 0.0

    test_acc = 100 * test_correct / test_total
    print(f"{model_name} Accuracy: {test_acc:.1f} %")
    return test_acc

def evaluate_segmentation(model, test_loader, device):
    print(f"\nEvaluating Segmentation Model")
    model.eval()
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
         for images, masks in test_loader:
             images, masks = images.to(device), masks.to(device)
             preds = model(images)
             
             total_dice += calculate_dice(preds, masks)
             total_pixel_acc += calculate_pixel_accuracy(preds, masks)
             num_batches += 1
             
    if num_batches == 0:
        print("No test data found for segmentation.")
        return 0.0, 0.0

    avg_dice = total_dice / num_batches
    avg_pixel_acc = total_pixel_acc / num_batches
    print(f"Segmentation Dice Score: {avg_dice:.2f}")
    return avg_dice, avg_pixel_acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate Accuracy for the Brain CT Project")
    
    # Classification paths
    parser.add_argument('--stroke_dir', type=str, help="Dataset folder for Stroke classification (must have 'test' subfolder)")
    parser.add_argument('--stroke_weights', type=str, default="stroke_classifier_weights.pth", help="Path to stroke weights")
    
    parser.add_argument('--type_dir', type=str, help="Dataset folder for Stroke Type classification (must have 'test' subfolder)")
    parser.add_argument('--type_weights', type=str, default="stroke_type_weights.pth", help="Path to type weights")
    
    # Segmentation paths
    parser.add_argument('--seg_images', type=str, help="Folder containing segmentation test images")
    parser.add_argument('--seg_masks', type=str, help="Folder containing segmentation test masks")
    parser.add_argument('--unet_weights', type=str, default="unet_weights.pth", help="Path to U-Net weights")
    
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    stroke_acc = None
    type_acc = None
    seg_dice = None

    # 1. Evaluate Stroke Classifier
    if args.stroke_dir and os.path.exists(args.stroke_dir):
        if os.path.exists(args.stroke_weights):
            _, _, test_loader, class_names = get_data_loaders(args.stroke_dir, batch_size=args.batch_size)
            model = StrokeClassifier(num_classes=len(class_names)).to(device)
            model.load_state_dict(torch.load(args.stroke_weights, map_location=device))
            stroke_acc = evaluate_classification(model, test_loader, device, "Stroke")
        else:
            print(f"⚠️ Stroke weights not found at {args.stroke_weights}")

    # 2. Evaluate Stroke Type Classifier
    if args.type_dir and os.path.exists(args.type_dir):
        if os.path.exists(args.type_weights):
            _, _, test_loader, class_names = get_data_loaders(args.type_dir, batch_size=args.batch_size)
            model = StrokeTypeClassifier(num_classes=len(class_names)).to(device)
            model.load_state_dict(torch.load(args.type_weights, map_location=device))
            type_acc = evaluate_classification(model, test_loader, device, "Stroke Type")
        else:
            print(f"⚠️ Type weights not found at {args.type_weights}")

    # 3. Evaluate U-Net Segmentation
    if args.seg_images and args.seg_masks:
        if os.path.exists(args.unet_weights):
            test_dataset = SegmentationDataset(args.seg_images, args.seg_masks)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size//2, shuffle=False)
            model = UNet(n_channels=3, n_classes=1).to(device)
            model.load_state_dict(torch.load(args.unet_weights, map_location=device))
            seg_dice, _ = evaluate_segmentation(model, test_loader, device)
        else:
            print(f"⚠️ U-Net weights not found at {args.unet_weights}")

    print("\nFINAL PROJECT PERFORMANCE")
    if stroke_acc is not None:
        print(f"Stroke Detection Accuracy : {stroke_acc:.1f} %")
    if type_acc is not None:
        print(f"Stroke Type Accuracy      : {type_acc:.1f} %")
    if seg_dice is not None:
        print(f"Segmentation Dice Score   : {seg_dice:.2f}")

if __name__ == "__main__":
    main()
