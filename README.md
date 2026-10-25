# Handwritten Digit Recognition: 0 vs 1

A beginner-friendly binary classification project that uses **TensorFlow/Keras** to recognize handwritten digits **0 and 1** from numerical image features stored in a CSV dataset.

## Overview

This project demonstrates a basic machine-learning workflow:

1. Load and preprocess the dataset.
2. Separate image features (`X`) from labels (`y`).
3. Split the data into training (80%) and testing (20%) sets.
4. Standardize the input features with `StandardScaler`.
5. Train a feed-forward neural network.
6. Evaluate the model using test loss and accuracy.
7. Compare predicted labels with the actual labels.

## Model Architecture

The neural network uses:

| Layer | Units | Activation |
|---|---:|---|
| Input | Number of dataset features | — |
| Hidden layer 1 | 25 | Sigmoid |
| Hidden layer 2 | 15 | Sigmoid |
| Output layer | 1 | Sigmoid |

- **Loss:** Binary cross-entropy
- **Optimizer:** Adam
- **Learning rate:** `0.01`
- **Training epochs:** `100`
- **Batch size:** `32`
- **Evaluation metric:** Accuracy

The output layer returns a probability. The project uses a threshold of `0.5` to classify each sample as digit `0` or digit `1`.

## Project Structure

Example structure:

```text
project/
├── dataset/
│   └── train.csv
├── pd.py
├── main.py
├── requirements.txt
└── README.md
```

- `dataset/train.csv` — input dataset. It should contain a `label` column and feature columns.
- `pd.py` — module containing the `preprocess()` function.
- `main.py` — loads the processed data, trains the model, and evaluates it.
- `requirements.txt` — Python dependencies.

Adjust the filenames in this example to match your project.

## Dataset Format

The CSV file should contain:
- A `label` column with values `0` or `1`.
- Other columns containing numerical features, such as image pixel values.

Example:

| pixel_1 | pixel_2 | pixel_3 | label |
|---:|---:|---:|---:|
| 0 | 255 | 0 | 0 |
| 255 | 0 | 255 | 1 |

The `preprocess()` function should return a Pandas DataFrame with the cleaned data and a `label` column.

## Requirements

- Python 3.10 or a compatible version supported by your TensorFlow installation
- NumPy
- Pandas
- Scikit-learn
- TensorFlow

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your repository URL and folder name.

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn tensorflow
```

Alternatively, create a `requirements.txt` containing:

```text
numpy
pandas
scikit-learn
tensorflow
```

Then run:

```bash
pip install -r requirements.txt
```

## Run the Project

Run the main Python file:

```bash
python main.py
```

Make sure:
- `main.py` imports `preprocess` from the correct module.
- The dataset exists at `dataset/train.csv`, or update the path in your preprocessing code.
- The dataset has a `label` column containing both classes, `0` and `1`.

## How It Works

### 1. Preprocessing

The `preprocess()` function loads and cleans the dataset. The main script separates features and labels:

```python
X = df.drop(columns=["label"])
y = df["label"]
```

### 2. Splitting the dataset

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=50,
    stratify=y
)
```

This reserves 80% of the data for training and 20% for testing. `stratify=y` helps preserve the class proportions in each split.

### 3. Feature scaling

`StandardScaler` is fitted on the training features only. The same transformation is applied to the test features to avoid data leakage.

### 4. Training

The model learns patterns in the training data over 100 epochs.

### 5. Evaluation and prediction

The model reports test loss and test accuracy. Predicted probabilities are converted into labels using a threshold of `0.5`.

## Notes and Limitations

- This is a simple fully connected neural network intended as a learning project.
- Actual accuracy depends on dataset quality, class balance, preprocessing, and model settings.
- The README does not claim a specific accuracy because results must be measured by running the model.
- For image data, a Convolutional Neural Network (CNN) may perform better by learning spatial patterns.
- Consider using a validation set and early stopping to monitor overfitting.
- If pixel values are raw grayscale intensities from `0` to `255`, dividing by `255.0` is another common baseline. Use one scaling strategy consistently.

## Future Improvements

- Add a validation set and early stopping.
- Display a confusion matrix and classification report.
- Plot training and validation loss/accuracy.
- Save the trained model and scaler.
- Build a function to predict a single uploaded image.
- Compare the current network with a CNN.

## License

Add a license file if you intend to publish or share this project publicly.
