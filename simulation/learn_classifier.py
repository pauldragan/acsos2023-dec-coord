import pandas as pd
import sklearn
import numpy as np

def main():
    df = pd.read_pickle("manager_satisfaction.pkl")
    print(df)

    df_grouped = df.groupby("ts")
    print(df_grouped)
    for ts, df_group in df_grouped:
        print(df_group)


if __name__ == "__main__":
    main()
