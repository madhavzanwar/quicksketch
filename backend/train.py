# backend/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from utils.dataset import get_dataloaders, CATEGORIES
from models.cnn import QuickDrawCNN

# --- Configuration ---
CONFIG = {
    'data_dir': os.path.join(os.path.dirname(__file__), 'data'),
    'model_save_dir': os.path.join(os.path.dirname(__file__), 'models'),
    'num_classes': len(CATEGORIES),
    'batch_size': 64,
    'max_samples_per_class': 5000,
    'learning_rate': 0.001,
    'num_epochs': 10,
}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients from previous step
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx + 1}/{len(loader)} | '
                  f'Loss: {running_loss / (batch_idx + 1):.4f} | '
                  f'Acc: {100. * correct / total:.1f}%')
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():     # no gradient computation during validation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Training on {CONFIG["num_classes"]} categories: {CATEGORIES}\n')
    
    # Data
    print('Loading data...')
    train_loader, val_loader = get_dataloaders(
        CONFIG['data_dir'],
        CATEGORIES,
        batch_size=CONFIG['batch_size'],
        max_samples_per_class=CONFIG['max_samples_per_class']
    )
    
    # Model
    model = QuickDrawCNN(num_classes=CONFIG['num_classes']).to(device)
    print(f'\nModel parameters: {model.count_parameters():,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print('\nStarting training...\n')
    
    for epoch in range(CONFIG['num_epochs']):
        print(f'Epoch {epoch + 1}/{CONFIG["num_epochs"]}')
        print('-' * 50)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%')
        print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(
                CONFIG['model_save_dir'], 'best_model.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'categories': CATEGORIES,
            }, model_path)
            print(f'  ✓ New best model saved (val_acc: {val_acc:.1f}%)')
        
        print()
    
    # Save training history
    history_path = os.path.join(CONFIG['model_save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    print(f'Training complete. Best validation accuracy: {best_val_acc:.1f}%')
    print(f'Model saved to: {CONFIG["model_save_dir"]}/best_model.pth')


if __name__ == '__main__':
    train()