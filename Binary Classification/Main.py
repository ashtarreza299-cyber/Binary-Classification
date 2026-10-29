
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Dense

from pd import preprocess

def main():

    # 1. Loading Dataset
    df = preprocess()
    print("Precessing DataSet...")

    X = df.drop(columns=['label'])
    Y = df['label']

    # 2. Splltting data set to training(80%) and testing set(20%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

    # 3. Normalization
    print("Normalizing DataSet...")
    scalar = StandardScaler()
    Xn_train = scalar.fit_transform(X_train)
    Xn_test = scalar.transform(X_test)


    # 4. Build a Neural Network
    tf.random.set_seed(1234)

    model = Sequential([

        Dense(25, activation='sigmoid', name='layer1'),
        Dense(15, activation='sigmoid', name='layer2'),
        Dense(1, activation='sigmoid', name='outputlayer')

    ])

    # 5. Compile the model
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"]
    )

    # 6. Train the model
    model.fit(Xn_train, Y_train, epochs=100, batch_size=32, verbose=1)

    # 7. Evaluate model on test data
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(
        Xn_test,
        Y_test,
        verbose=0
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")

    # 8. Predict test images
    probabilities = model.predict(Xn_test, verbose=0)
    predictions = (probabilities >= 0.5).astype(int).flatten()

    print("\nFirst 10 predictions:")
    print(predictions[:10])

    print("\nFirst 10 actual labels:")
    print(Y_test.iloc[:10].to_numpy())

if __name__ == "__main__":

    main()