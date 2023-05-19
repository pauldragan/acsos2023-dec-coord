import sys
import argparse
from jobs import JobReader, RefJobReader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("input_file", type=str, help="Path to the input .csv or .csv.gz file with jobs log.")
    parser.add_argument("manager_logs", type=str, help="Path to the manager logs .pkl.")
    args = parser.parse_args()

    df_managers = pd.read_pickle(args.manager_logs)
    # df_managers = df_managers.set_index("ts")
    print(df_managers)

    df_diff = df_managers[df_managers["real_policy"] != df_managers["selected_policy"]]
    print(df_diff)

    df_managers["real_policy"] = df_managers["real_policy"].astype("category")
    df_managers[["real_policy"]] = df_managers[["real_policy"]].apply(lambda x: x.cat.codes)
    df_managers.groupby("name").plot(x="ts", y="real_policy", legend=True)
    plt.show()



    # df_managers_groups = df_managers.groupby("name")
    # for dfman_name, dfman in df_managers_groups:
    #     dfman["real_policy"] = dfman["real_policy"].astype("category")
    #     dfman[["real_policy"]] = dfman[["real_policy"]].apply(lambda x: x.cat.codes)
    #     print(dfman)
    #     dfman.plot(x="ts", y="real_policy")
    #     plt.show()

    # df_managers.plot(x="ts", y="real_policy")
    # plt.show()

    # df = pd.read_csv(args.input_file, sep=";", header=0)
    # print(df)
    # print("Unique groups: ", df["group_id"].nunique())
    # print("Unique top-level groups: ", df["tlgroup_id"].nunique())
    # print("Unique runtimes: ", df["runtime_id"].nunique())
    # print(df.groupby(["tlgroup_id"])["submission_id"].count())
    # print(df.groupby(["runtime_id"])["submission_id"].count())

    # df["spawn_datetime"] = pd.to_datetime(df["spawn_ts"], unit="s")
    # print(df["spawn_ts"])
    # print(df["spawn_datetime"])

    # # hist = df.hist(column="spawn_datetime", bins=1500)
    # # hist = df.hist(column="spawn_ts", bins=1500)

    # energy_cost = 10 * (df["spawn_ts"] / df["spawn_ts"].max()) + np.sin(df["spawn_ts"] / 5000000)
    # plt.plot(df["spawn_ts"], energy_cost)
    # plt.show()

if __name__ == "__main__":
    main()
