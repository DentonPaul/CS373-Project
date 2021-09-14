import pandas as pd
import numpy as np


def extract(print_info):

    # Import csv to Dataframe

    df = pd.read_csv('../dataset/wine-quality-red.csv')

    if print_info:

        # Identify Sample Size

        print("Number of samples: ")
        print(str(len(df)))
        print()

    # Extract y dataset and convert to binary

    quality_data = df.iloc[:, 11].to_numpy()
    y = np.where(quality_data >= 7, 1, 0)
    if print_info:
        print("The y dataset:")
        print(y)
        print()

    # Extract X dataset

    X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].to_numpy()
    if print_info:
        print("The X dataset:")
        print(X)
        print()

    return X, y
