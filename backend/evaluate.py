import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

from predict import load_model
from train import CONFIG, set_random_seed
from utils.dataset import CATEGORIES, get_dataloaders


MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')
SUMMARY_PATH = os.path.join(MODELS_DIR, 'training_summary.json')

CONFUSION_MATRIX_PNG = os.path.join(MODELS_DIR, 'confusion_matrix.png')
CONFUSION_MATRIX_JSON = os.path.join(MODELS_DIR, 'confusion_matrix.json')
CLASS_METRICS_JSON = os.path.join(MODELS_DIR, 'class_metrics.json')
CONFUSION_PAIRS_JSON = os.path.join(MODELS_DIR, 'confusion_pairs.json')


def load_random_seed():
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        return int(summary.get('random_seed', CONFIG['random_seed']))
    return int(CONFIG['random_seed'])


def get_validation_loader(random_seed):
    _, val_loader, class_summary = get_dataloaders(
        CONFIG['data_dir'],
        CATEGORIES,
        batch_size=CONFIG['batch_size'],
        max_samples_per_class=CONFIG['max_samples_per_class'],
        train_ratio=CONFIG['train_ratio'],
        random_seed=random_seed,
    )
    return val_loader, class_summary


def collect_predictions(model, device, loader):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.tolist())

    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def build_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        matrix[true_idx, pred_idx] += 1
    return matrix


def compute_class_metrics(confusion_matrix, class_names):
    metrics = []

    for idx, class_name in enumerate(class_names):
        tp = int(confusion_matrix[idx, idx])
        fp = int(confusion_matrix[:, idx].sum() - tp)
        fn = int(confusion_matrix[idx, :].sum() - tp)
        support = int(confusion_matrix[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        metrics.append({
            'class_name': class_name,
            'label_index': idx,
            'precision': round(precision, 6),
            'recall': round(recall, 6),
            'f1_score': round(f1, 6),
            'support': support,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
        })

    return metrics


def compute_top_confusion_pairs(confusion_matrix, class_names):
    pairs = []

    for true_idx, true_name in enumerate(class_names):
        for pred_idx, pred_name in enumerate(class_names):
            if true_idx == pred_idx:
                continue

            count = int(confusion_matrix[true_idx, pred_idx])
            if count == 0:
                continue

            pairs.append({
                'true_class': true_name,
                'true_index': true_idx,
                'predicted_class': pred_name,
                'predicted_index': pred_idx,
                'count': count,
            })

    pairs.sort(
        key=lambda item: (
            -item['count'],
            item['true_class'],
            item['predicted_class'],
        )
    )
    return pairs


def save_confusion_matrix_plot(confusion_matrix, class_names, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(confusion_matrix, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title('QuickSketch Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    threshold = confusion_matrix.max() / 2 if confusion_matrix.size else 0
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            value = int(confusion_matrix[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha='center',
                va='center',
                color='white' if value > threshold else '#1A1A1A',
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model checkpoint not found: {MODEL_PATH}')

    random_seed = load_random_seed()
    set_random_seed(random_seed)

    model, device = load_model(MODEL_PATH)
    val_loader, class_summary = get_validation_loader(random_seed)

    y_true, y_pred = collect_predictions(model, device, val_loader)
    confusion_matrix = build_confusion_matrix(y_true, y_pred, len(CATEGORIES))
    class_metrics = compute_class_metrics(confusion_matrix, CATEGORIES)
    confusion_pairs = compute_top_confusion_pairs(confusion_matrix, CATEGORIES)

    total_samples = int(confusion_matrix.sum())
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    confusion_matrix_payload = {
        'class_names': CATEGORIES,
        'num_classes': len(CATEGORIES),
        'total_samples': total_samples,
        'accuracy': round(accuracy, 6),
        'random_seed': random_seed,
        'validation_split': {
            'train_ratio': CONFIG['train_ratio'],
            'validation_ratio': round(1 - CONFIG['train_ratio'], 6),
            'batch_size': CONFIG['batch_size'],
            'max_samples_per_class': CONFIG['max_samples_per_class'],
        },
        'class_summary': class_summary,
        'matrix': confusion_matrix.tolist(),
    }

    class_metrics_payload = {
        'class_names': CATEGORIES,
        'num_classes': len(CATEGORIES),
        'total_samples': total_samples,
        'metrics': class_metrics,
    }

    confusion_pairs_payload = {
        'class_names': CATEGORIES,
        'num_classes': len(CATEGORIES),
        'total_pairs': len(confusion_pairs),
        'pairs': confusion_pairs,
    }

    save_json(CONFUSION_MATRIX_JSON, confusion_matrix_payload)
    save_json(CLASS_METRICS_JSON, class_metrics_payload)
    save_json(CONFUSION_PAIRS_JSON, confusion_pairs_payload)
    save_confusion_matrix_plot(confusion_matrix, CATEGORIES, CONFUSION_MATRIX_PNG)

    print('Evaluation complete.')
    print(f'Validation samples: {total_samples}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Saved: {CONFUSION_MATRIX_PNG}')
    print(f'Saved: {CONFUSION_MATRIX_JSON}')
    print(f'Saved: {CLASS_METRICS_JSON}')
    print(f'Saved: {CONFUSION_PAIRS_JSON}')


if __name__ == '__main__':
    main()
