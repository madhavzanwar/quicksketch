# backend/train.py

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import QuickDrawCNN
from utils.dataset import get_dataloaders, CATEGORIES

# --- Configuration ---
CONFIG = {
    'data_dir': os.path.join(os.path.dirname(__file__), 'data'),
    'model_save_dir': os.path.join(os.path.dirname(__file__), 'models'),
    'num_classes': len(CATEGORIES),
    'batch_size': 64,
    'max_samples_per_class': 5000,
    'train_ratio': 0.8,
    'learning_rate': 0.001,
    'num_epochs': 20,
    'early_stopping_patience': 5,
    'random_seed': 42,
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_data_summary(class_summary, random_seed):
    print(f'Random seed: {random_seed}')
    print('Samples per class:')
    for category, stats in class_summary.items():
        print(
            f'  {category}: sampled={stats["sampled"]} | '
            f'train={stats["train"]} | val={stats["val"]}'
        )
    print()

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
    set_random_seed(CONFIG['random_seed'])

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Training on {CONFIG["num_classes"]} categories: {CATEGORIES}\n')
    
    # Data
    print('Loading data...')
    train_loader, val_loader, class_summary = get_dataloaders(
        CONFIG['data_dir'],
        CATEGORIES,
        batch_size=CONFIG['batch_size'],
        max_samples_per_class=CONFIG['max_samples_per_class'],
        train_ratio=CONFIG['train_ratio'],
        random_seed=CONFIG['random_seed']
    )
    print_data_summary(class_summary, CONFIG['random_seed'])
    
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
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
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
            best_epoch = epoch + 1
            epochs_without_improvement = 0
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
            print(f'  New best model saved (val_acc: {val_acc:.1f}%)')
        else:
            epochs_without_improvement += 1
            print(
                '  No validation accuracy improvement '
                f'({epochs_without_improvement}/'
                f'{CONFIG["early_stopping_patience"]})'
            )

            if epochs_without_improvement >= CONFIG['early_stopping_patience']:
                early_stopped = True
                print('  Early stopping triggered.')
                print()
                break

        print()
    
    # Save training history
    history_path = os.path.join(CONFIG['model_save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

    training_summary = {
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'num_epochs_run': len(history['train_loss']),
        'early_stopped': early_stopped,
        'random_seed': CONFIG['random_seed'],
    }
    summary_path = os.path.join(CONFIG['model_save_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f)

    print(f'Training complete. Best validation accuracy: {best_val_acc:.1f}%')
    print(f'Best epoch: {best_epoch}')
    print(f'Epochs run: {len(history["train_loss"])}')
    print(f'Early stopped: {early_stopped}')
    print(f'Model saved to: {CONFIG["model_save_dir"]}/best_model.pth')


if __name__ == '__main__':
    train()
