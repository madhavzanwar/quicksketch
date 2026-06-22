import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CATEGORIES = [
    'apple',
    'bicycle',
    'book',
    'camera',
    'chair',
    'crown',
    'diamond',
    'fish',
    'guitar',
    'house',
    'lightning',
    'mountain',
    'pizza',
    'star',
    'sun',
    'umbrella'
]


class QuickDrawDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = (
            self.data[idx]
            .astype(np.float32)
            / 255.0
        )

        image = image.reshape(
            1,
            28,
            28
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return image, label


def get_dataloaders(
    data_dir,
    categories,
    batch_size=64,
    max_samples_per_class=5000,
    train_ratio=0.8,
    random_seed=42
):
    rng = np.random.default_rng(random_seed)

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    class_summary = {}

    for label_idx, category in enumerate(categories):
        file_path = os.path.join(data_dir, f"{category}.npy")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find: {file_path}")

        raw = np.load(file_path)
        sample_count = min(max_samples_per_class, len(raw))

        if len(raw) > sample_count:
            sampled_indices = rng.choice(len(raw), size=sample_count, replace=False)
            sampled = raw[sampled_indices]
        else:
            sampled = raw.copy()

        sampled = sampled[rng.permutation(len(sampled))]
        split_idx = int(len(sampled) * train_ratio)

        train_split = sampled[:split_idx]
        val_split = sampled[split_idx:]

        train_data.append(train_split)
        val_data.append(val_split)
        train_labels.extend([label_idx] * len(train_split))
        val_labels.extend([label_idx] * len(val_split))

        class_summary[category] = {
            'available': int(len(raw)),
            'sampled': int(len(sampled)),
            'train': int(len(train_split)),
            'val': int(len(val_split)),
        }

    train_dataset = QuickDrawDataset(
        data=np.concatenate(train_data, axis=0),
        labels=train_labels
    )

    val_dataset = QuickDrawDataset(
        data=np.concatenate(val_data, axis=0),
        labels=val_labels
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, class_summary


if __name__ == "__main__":

    DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data"
        )
    )

  
    print(f"Data Directory: {DATA_DIR}")

    train_loader, val_loader, class_summary = get_dataloaders(
        DATA_DIR,
        CATEGORIES,
        batch_size=64,
        max_samples_per_class=5000
    )

    print("\nSampling Summary")
    print("-" * 40)
    for category, stats in class_summary.items():
        print(
            f"{category}: sampled={stats['sampled']} | "
            f"train={stats['train']} | val={stats['val']}"
        )

    images, labels = next(
        iter(train_loader)
    )

    print("\nBatch Inspection")
    print("-" * 40)
    print(
        f"Images Shape: {images.shape}"
    )
    print(
        f"Labels Shape: {labels.shape}"
    )
    print(
        f"Images Type: {images.dtype}"
    )
    print(
        f"Labels Type: {labels.dtype}"
    )
    print(
        f"Pixel Range: "
        f"[{images.min():.3f}, "
        f"{images.max():.3f}]"
    )

    print(
        f"First 10 Labels: "
        f"{labels[:10]}"
    )

    print(
        "Category Names:"
    )

    print(
        [
            CATEGORIES[
                label.item()
            ]
            for label in labels[:10]
        ]
    )
