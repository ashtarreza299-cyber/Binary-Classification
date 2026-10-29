# Binary and Multiclass Classification

This repository contains two small machine-learning projects built with
Python and TensorFlow:

-   **Binary Classification** --- trains a neural network to predict one
    of two classes (`0` or `1`) from tabular data.
-   **Multi Class Classification** --- trains a neural network to
    classify handwritten digit images into ten classes (`0`--`9`), using
    flattened pixel features from a CSV dataset. It also includes a
    script to predict a digit from an image file.

These projects are intended for learning and experimentation with data
preprocessing, neural networks, model evaluation, and prediction.

## Project Structure

``` text
Binary Classification/
├── Main.py                 # Train and evaluate the binary classifier
├── pd.py                   # Load and clean the binary-classification dataset
├── README.md               # Documentation for this repository
└── dataset/
    ├── train.csv           # Training data used by the scripts
    ├── test.csv             # Additional test data (not currently used by Main.py)
    └── sample_submission.csv

Multi Class Classification/
├── ml.py                   # Train, evaluate, and save the digit classifier
├── pd.py                   # Load and clean the multiclass dataset
├── predict_digit.py        # Predict a digit from an image
├── digit_model.keras       # Saved Keras model (created by ml.py)
└── 5.png                   # Example digit image for prediction
```

> The layout above follows the files shown in the project screenshot.
> Keep each script in its current folder unless you also update its
> relative file paths.

## Features

### 1. Binary classification

The binary-classification pipeline:

1.  Loads `dataset/train.csv` and removes rows containing missing
    values.
2.  Keeps only rows whose `label` is `0` or `1`.
3.  Separates input features from the `label` column.
4.  Splits the data into training (80%) and testing (20%) subsets.
5.  Standardizes features with `StandardScaler`.
6.  Trains a dense neural network with two hidden layers and a sigmoid
    output.
7.  Evaluates test loss and accuracy.
8.  Prints the first ten predictions alongside their actual labels.

**Model configuration**

-   Hidden layers: 25 and 15 neurons, each using sigmoid activation
-   Output layer: 1 neuron with sigmoid activation
-   Loss: binary cross-entropy
-   Optimizer: Adam with learning rate `0.01`
-   Training: 100 epochs, batch size 32
-   Train/test split: 80/20, with `random_state=50`

### 2. Multiclass digit classification

The multiclass pipeline:

1.  Loads `../dataset/train.csv` relative to the
    `Multi Class Classification` folder.
2.  Removes rows containing missing values.
3.  Separates pixel features from the `label` column.
4.  Converts pixel features to `float32` and scales values from
    `[0, 255]` to `[0, 1]`.
5.  Splits the data into training (80%) and testing (20%) subsets.
6.  Trains a neural network to predict one of ten digit classes.
7.  Saves the trained model as `digit_model.keras`.
8.  Reports test loss, accuracy, a classification report, and a
    confusion matrix.

**Model configuration**

-   Hidden layers: 25 and 15 neurons, using ReLU activation
-   Output layer: 10 neurons with softmax activation
-   Loss: sparse categorical cross-entropy
-   Optimizer: Adam
-   Training: 10 epochs
-   Train/test split: 80/20, with `random_state=50`

The dataset is expected to contain a `label` column and pixel-feature
columns. For the current model, each row should contain **784 pixel
features** representing a 28 × 28 grayscale image, plus the label.

### 3. Predict a digit from an image

`predict_digit.py` loads `digit_model.keras`, opens an image, converts
it to grayscale, resizes it to 28 × 28 pixels, adjusts polarity when the
background appears bright, scales pixel values, and prints the predicted
digit and its confidence.

The script currently uses `5.png` as its example input. Change the
`image_path` value in the final block of `predict_digit.py` to use a
different image.

For best results, use a clear image containing one centered digit with a
plain background. The preprocessing in this script is a simple
heuristic; images that differ substantially from the training data may
be classified incorrectly.

## Requirements

Use Python 3.10 or a compatible version supported by your installed
TensorFlow release.

Required libraries:

-   NumPy
-   pandas
-   scikit-learn
-   TensorFlow
-   Pillow

Install the dependencies in a virtual environment:

``` bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

``` powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

Install packages:

``` bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn tensorflow pillow
```

## Dataset Setup

The scripts expect a CSV file named `train.csv` with a column named
`label`.

-   **Binary classification:** place the file at
    `Binary Classification/dataset/train.csv`.
-   **Multiclass classification:** the current
    `Multi Class Classification/pd.py` reads `../dataset/train.csv`,
    which resolves to a `dataset` directory one level above
    `Multi Class Classification`. With the screenshot's structure, this
    means the shared `dataset/train.csv` at the repository root level.

Make sure the CSV format matches the selected task: - For binary
classification, labels must be `0` or `1`; other labels are filtered
out. - For digit classification, labels should be integers from `0` to
`9`, and the remaining columns should be pixel values.

If your folders are arranged differently, update the paths in the
corresponding `pd.py` file. Running scripts from the expected working
directory is important because the paths are relative.

## How to Run

### Binary classifier

Open a terminal in the `Binary Classification` folder and run:

``` bash
python Main.py
```

This script imports `preprocess` from the local `pd.py` and reads
`dataset/train.csv` relative to the current working directory.

### Train the multiclass digit model

Open a terminal in the `Multi Class Classification` folder and run:

``` bash
python ml.py
```

After training completes, the script saves or overwrites
`digit_model.keras` in that folder.

### Predict a digit

From the `Multi Class Classification` folder, make sure
`digit_model.keras` and the chosen image are present, then run:

``` bash
python predict_digit.py
```

By default, this predicts the image at `5.png`.

## Evaluation Output

The training scripts print evaluation metrics to the terminal:

-   **Loss:** how far model predictions are from the expected labels
    according to the selected loss function.
-   **Accuracy:** the fraction of examples classified correctly.
-   **Classification report:** precision, recall, and F1-score for each
    digit class (multiclass project).
-   **Confusion matrix:** counts of actual-versus-predicted classes
    (multiclass project).

Results depend on the dataset, data quality, preprocessing, and training
configuration. No particular accuracy is guaranteed.

## Important Notes and Potential Improvements

-   **Fit preprocessing only on training data.** The binary script
    correctly fits `StandardScaler` on `X_train` and applies it to
    `X_test`. Save the fitted scaler alongside the model if you later
    need consistent predictions on new binary-classification records.
-   **Keep feature order consistent.** Prediction inputs must use the
    same feature columns and order as training.
-   **Consider stratified splitting.** For imbalanced classes, use
    `train_test_split(..., stratify=Y, ...)` when every class has enough
    examples.
-   **Use reproducible evaluation.** Set seeds and record the dataset
    version and model configuration.
-   **Avoid evaluating on training data.** Use the held-out test set for
    evaluation, as the scripts currently do.
-   **Use a stable preprocessing path.** Relative paths depend on the
    working directory; using paths based on `__file__` can make scripts
    more reliable.
-   **Check the image input shape.** The multiclass model is trained on
    784 flat pixel features. The prediction script reshapes input
    according to `model.input_shape`; the flattened-input path matches
    this model.
-   **Model file location matters.** `predict_digit.py` loads
    `digit_model.keras` relative to the current working directory, so
    run it from the `Multi Class Classification` folder.
-   **The binary model is not saved currently.** `Main.py` trains and
    evaluates the binary classifier each time it runs.

## Troubleshooting

**`FileNotFoundError` for `train.csv`** - Check that the dataset file
exists at the path expected by the relevant `pd.py`. - Run the command
from the documented folder, or update the relative path.

**`ModuleNotFoundError`** - Activate the virtual environment and install
the required dependencies.

**Unexpected input-shape errors** - For digit classification, confirm
there are 784 pixel columns in addition to `label`. - Ensure the
training data uses the same feature layout as the images processed by
`predict_digit.py`.

**Poor digit predictions** - Use images that resemble the training data
in size, contrast, digit placement, and background. - Remember that
resizing and simple inversion may not reproduce the exact preprocessing
used to create the training CSV.

## License

No license is specified in this repository. Add a `LICENSE` file before
distributing the project if you want to define how others may use,
modify, or share it.
