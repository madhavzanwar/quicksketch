# QuickSketch

AI-powered sketch recognition system trained on Google's Quick Draw dataset.

## Features

- Download Quick Draw datasets automatically
- Explore and visualize sketch samples
- Train neural network models
- Predict hand-drawn sketches

## Tech Stack

- Python
- NumPy
- Matplotlib
- PyTorch (coming soon)

## Dataset

Google Quick Draw Dataset

Dataset files are not stored in the repository. Run:

```bash
python backend/utils/download_data.py
```

to download the required categories.

## Project Structure

```text
backend/
├── data/
├── models/
├── utils/
│   ├── download_data.py
│   └── explore_data.py
├── train.py
├── predict.py
└── app.py
```