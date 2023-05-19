import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("../data/release01-2021-12-29/data.csv.gz", sep=";", header=0)

   # spawn_ts = df["spawn_ts"]
    firstVal = df["spawn_ts"][0]
    firstVal = 0

    timePassed = []
    for val in df["spawn_ts"]:
        timePassed.append(val - firstVal)

    df = pd.DataFrame(timePassed, columns=['time'])
    sin= np.sin(df / 9_000_000)
    print(df)

    plt.plot(df, sin, label='cost')
    cap = 0.8
    satis = sin.copy(True)
    satis = (1 - satis)
    satis[abs(satis) > cap] = cap

#    for idx, val in enumerate(satis['time']):
#        if val == 'time':
#            continue
#        if val >= cap:
#            satis['time'].replace([val], [cap])
        #satis[val] = val if 1 - (int(val)) <= cap else cap
    #satis = 1-sin if 1-sin < cap else cap
    plt.plot(df, satis, label='satisfaction')
    plt.legend()
    plt.show()

    print(sin)
    print(satis)
    #plt.savefig(fname="../sinfct.png")

    #print(spawn_ts[0])

    #print(df["spawn_ts"])


if __name__ == "__main__":
    main()