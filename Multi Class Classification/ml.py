import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Sequential
from pd import preprocess

def main():

    # 1. Loading dataset
    print("Loading Data set...")
    dataset = preprocess()

    # 2. Split the data set to training set (80%) and testing set (20%)

    X = dataset.drop(columns=["label"]).astype("float32") / 255.0
    Y = dataset["label"]

    print("Number of pixel features:", X.shape[1])
    print("Pixel minimum:", X.min().min())
    print("Pixel maximum:", X.max().max())
    print("First few feature names:", list(X.columns[:10]))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=50
    )

    # 3. Create Model
    tf.random.set_seed(1234)
    model = Sequential ([
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(10, activation='softmax'),
    ])

    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    model.fit(X_train, Y_train, epochs=10)

    # save the model
    model.save("digit_model.keras")

    # 4. Prediction and Evaluation
    # Predict class probabilities
    predictions = model.predict(X_test)

    # Convert probabilities into predicted class labels
    Y_pred = np.argmax(predictions, axis=1)

    print("\n\n-----------------------------------------------\n\n")
    print("\nPredicted labels:")
    print(Y_pred)

    print("\nActual labels:")
    print(np.array(Y_test))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Additional evaluation metrics
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))

if __name__ == "__main__":

    main()