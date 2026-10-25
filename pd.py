import pandas as pd

def preprocess():


    # 1. Load dataset
    print("Loading Data Set...")
    df = pd.read_csv("dataset/train.csv")

    # 3. Remove rows with missing values
    df.dropna(inplace=True)

    # 4. Keep only labels 0 and 1
    df = df[df["label"].isin([0, 1])].copy()

    return df


if __name__ == "__main__":

    df = preprocess()