import pandas as pd


def preprocess():

    df = pd.read_csv("../dataset/train.csv")

    df.dropna(inplace=True)

    print(df['label'].unique())

    return df

if __name__ == "__main__":

    df = preprocess()