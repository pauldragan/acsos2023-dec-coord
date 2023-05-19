import sys
import argparse
from jobs import JobReader, RefJobReader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input .csv or .csv.gz file with jobs log.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep=";", header=0)

    print(df)
    print("Unique groups: ", df["group_id"].nunique())
    print("Unique top-level groups: ", df["tlgroup_id"].nunique())
    print("Unique runtimes: ", df["runtime_id"].nunique())
    print(df.groupby(["tlgroup_id"])["submission_id"].count())
    print(df.groupby(["runtime_id"])["submission_id"].count())

    df["spawn_datetime"] = pd.to_datetime(df["spawn_ts"], unit="s")
    print(df["spawn_ts"])
    print(df["spawn_datetime"])

    # hist = df.hist(column="spawn_datetime", bins=1500)
    # hist = df.hist(column="spawn_ts", bins=1500)

    TIMEFRAME = "15D"
    df_resampled = df[["spawn_datetime",
                       "submission_id"]].resample(TIMEFRAME,
                                                  axis=0,
                                                  on="spawn_datetime",
                                                  origin=df["spawn_datetime"][0]).count()
    df_resampled.plot()

    # energy_cost = 10 * (df["spawn_ts"] / df["spawn_ts"].max()) + np.sin(df["spawn_ts"] / 5000000)
    # plt.plot(df["spawn_ts"], energy_cost)
    plt.show()

if __name__ == "__main__":
    main()
