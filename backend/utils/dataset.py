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
    def __init__(
        self,
        data_dir,
        categories,
        max_samples_per_class=5000,
        split='train',
        train_ratio=0.8
    ):
        self.data = []
        self.labels = []

        for label_idx, category in enumerate(categories):

            file_path = os.path.join(
                data_dir,
                f"{category}.npy"
            )

            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Could not find: {file_path}"
                )

            raw = np.load(file_path)

            # Limit number of samples
            raw = raw[:max_samples_per_class]

            split_idx = int(len(raw) * train_ratio)

            if split == 'train':
                raw = raw[:split_idx]
            else:
                raw = raw[split_idx:]

            self.data.append(raw)
            self.labels.extend(
                [label_idx] * len(raw)
            )

            print(
                f"Loaded {category}: "
                f"{len(raw)} {split} samples"
            )

        self.data = np.concatenate(
            self.data,
            axis=0
        )

        self.labels = np.array(
            self.labels,
            dtype=np.int64
        )

        print(
            f"\n{split.upper()} SET: "
            f"{len(self.data)} samples"
        )

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
    max_samples_per_class=5000
):

    train_dataset = QuickDrawDataset(
        data_dir=data_dir,
        categories=categories,
        max_samples_per_class=max_samples_per_class,
        split='train'
    )

    val_dataset = QuickDrawDataset(
        data_dir=data_dir,
        categories=categories,
        max_samples_per_class=max_samples_per_class,
        split='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


if __name__ == "__main__":

    DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data"
        )
    )

  
    print(f"Data Directory: {DATA_DIR}")

    train_loader, val_loader = get_dataloaders(
        DATA_DIR,
        CATEGORIES,
        batch_size=64,
        max_samples_per_class=5000
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