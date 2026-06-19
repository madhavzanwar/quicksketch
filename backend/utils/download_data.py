import urllib.request
import os

categories = [
    'apple', 'bicycle', 'camera', 'sun',
    'pizza', 'umbrella', 'star', 'book',
    'fish', 'crown', 'diamond', 'lightning',
    'guitar', 'house', 'mountain', 'chair'
]

save_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(save_dir, exist_ok=True)

base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

for category in categories:
    filename = f'{category}.npy'
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f'Already exists: {filename}')
        continue

    url = base_url + filename.replace(' ', '%20')
    print(f'Downloading {filename}...')

    urllib.request.urlretrieve(url, save_path)

    print(f'Saved: {filename}')

print('All downloads complete.')