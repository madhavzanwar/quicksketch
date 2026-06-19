import numpy as np
import matplotlib.pyplot as plt
import os

categories = [
    'apple', 'bicycle', 'camera', 'sun',
    'pizza', 'umbrella', 'star', 'book',
    'fish', 'crown', 'diamond', 'lightning',
    'guitar', 'house', 'mountain', 'chair'
]

data_dir = os.path.join(os.path.dirname(__file__),'..','data')

sample_category = 'apple'
data = np.load(os.path.join(data_dir, f'{sample_category}.npy'))

print(f'category: {sample_category}')
print(f'shape: {data.shape}')
print(f'Data type: {data.dtype}')
print(f'Min value: {data.min()}, Max value: {data.max()}')
print(f'Total drawings: {data.shape[0]}')

# --- Step 2: Visualize 10 random drawings from one category ---
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle(f'Sample drawings: {sample_category}', fontsize=14)

indices = np.random.choice(len(data), 10, replace=False)
for i, idx in enumerate(indices):
    ax = axes[i // 4][i % 4]
    # reshape from flat 784 to 28x28
    img = data[idx].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_drawings.png', dpi=150)
plt.show()
print('Saved: sample_drawings.png')

# --- Step 3: Visualize one drawing from each category ---
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle('One drawing per category', fontsize=14)

for i, category in enumerate(categories):
    path = os.path.join(data_dir, f'{category}.npy')
    if not os.path.exists(path):
        print(f'Not yet downloaded: {category}')
        continue
    
    cat_data = np.load(path)
    ax = axes[i // 5][i % 5]
    img = cat_data[0].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(category, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('all_categories.png', dpi=150)
plt.show()
print('Saved: all_categories.png')

# --- Step 4: Dataset size summary ---
print('\n--- Dataset Summary ---')
for category in categories:
    path = os.path.join(data_dir, f'{category}.npy')
    if os.path.exists(path):
        d = np.load(path)
        print(f'{category:15s}: {d.shape[0]:,} drawings')
    else:
        print(f'{category:15s}: not downloaded yet')