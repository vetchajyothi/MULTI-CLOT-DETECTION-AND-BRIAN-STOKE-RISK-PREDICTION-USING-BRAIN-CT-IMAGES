import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from src.data_loader import get_data_loaders
from src.models.classification import StrokeClassifier, StrokeTypeClassifier

def train_classification_model(model_choice: str, data_dir: str, epochs: int, batch_size: int, learning_rate: float, save_path: str):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    print(f"Loading dataset from: {data_dir} for task: {model_choice}")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    print(f"Detected classes: {class_names}")
    num_classes = len(class_names)
    
    # Save class labels
    import json
    with open(save_path + "_classes.json", "w") as f:
        json.dump(class_names, f)
    
    # Initialize Model
    if model_choice.lower() == 'stroke':
        print("Initializing StrokeClassifier...")
        model = StrokeClassifier(num_classes=num_classes)
    elif model_choice.lower() == 'type':
        print("Initializing StrokeTypeClassifier...")
        model = StrokeTypeClassifier(num_classes=num_classes)
        
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🔥 Saving best model to {save_path}")
            torch.save(model.state_dict(), save_path)

    print("\n--- Training Complete ---")

    # ✅ IMPORTANT FIX: force save final model
    torch.save(model.state_dict(), save_path)
    print(f"✅ Final model saved at: {save_path}")

    # Test Evaluation
    model.load_state_dict(torch.load(save_path))
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

    test_acc = 100 * test_correct / test_total
    print(f"🏆 Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['stroke', 'type'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    
    args = parser.parse_args()
    
    train_classification_model(
        model_choice=args.task,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
